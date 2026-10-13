import unittest

from hdepic_eval.extraction import extract_field, extract_next
from hdepic_eval.mapper import ActionMapper


class EvaluationExtractionTests(unittest.TestCase):
    def test_unclosed_current_and_nested_future_are_independent(self):
        raw = '<action>open cupboard\n<goal>prepare coffee<next><action>pick up mug</action></next>'
        self.assertEqual(extract_field(raw, 'action')['text'], 'open cupboard')
        self.assertEqual(extract_field(raw, 'goal')['text'], 'prepare coffee')
        self.assertEqual(extract_next(raw)['actions'], ['pick up mug'])

    def test_duplicate_or_crossed_field_is_ambiguous(self):
        for raw in ('<goal>x</goal><goal>y</goal>', '<goal>x<why>y</goal>', '</goal><goal>x'):
            self.assertIsNone(extract_field(raw, 'goal')['text'])

    def test_case_insensitive_labels_json_and_entities(self):
        self.assertEqual(extract_field('<GOAL>Wash &amp; dry</GOAL>', 'goal')['text'], 'Wash & dry')
        self.assertEqual(extract_field('WHY: make room\nGOAL: clean kitchen\nNEXT: wash bowl', 'why')['text'], 'make room')
        self.assertEqual(extract_next('{"next":["take bowl",null,"put bowl"]}')['actions'], ['take bowl', None, 'put bowl'])

    def test_duplicate_json_keys_never_choose_last_answer(self):
        self.assertIsNone(extract_field('{"goal":"x","goal":"y"}', 'goal')['text'])
        self.assertEqual(extract_next('{"next":["x"],"next":["y"]}')['actions'], [])

    def test_missing_explicit_slots_never_shift(self):
        cases = ('<next>a;;c</next>', '<next>a</next><next></next><next>c</next>',
                 '<next>["a", "", "c"]</next>', '<next>1. a\n2.\n3. c</next>',
                 '<next>1. a\n3. c</next>', '<next><action>a</action><action></action><action>c</action></next>')
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(extract_next(raw)['actions'], ['a', None, 'c'])

    def test_decimal_quantity_and_timestamp_not_numbered_slots(self):
        raw = '<next>take 1.5 tablespoons; wash bowl; put bowl</next>'
        self.assertEqual(extract_next(raw)['actions'], ['take 1.5 tablespoons', 'wash bowl', 'put bowl'])
        raw = '<next>[0.00, 1.00] take bowl, [1.00, 2.00] wash bowl, [2.00, 3.00] put bowl</next>'
        self.assertEqual(extract_next(raw)['total_actions'], 3)
        self.assertEqual(extract_next(raw)['representations'], ['timestamped_list'])

    def test_comma_list_requires_action_bearing_items(self):
        mapper = ActionMapper({'open': 1, 'pick up': 2, 'close': 3}, {'cupboard': 1, 'mug': 2})
        detector = lambda text: mapper.map_text(text)[0] is not None
        self.assertEqual(extract_next('<next>open cupboard, pick up mug, close cupboard</next>', detector)['total_actions'], 3)
        self.assertEqual(extract_next('<next>mix flour, sugar, eggs</next>', detector)['total_actions'], 1)

    def test_extras_preserved_for_metric_diagnostics(self):
        self.assertEqual(extract_next('<next>a;b;c;d;</next>')['actions'], ['a', 'b', 'c', 'd'])
        self.assertEqual(extract_next('<next>1. a 2. b 3. c</next>')['actions'], ['a', 'b', 'c'])


class EvaluationMapperTests(unittest.TestCase):
    def setUp(self):
        self.mapper = ActionMapper({'pick up': 0, 'place': 1, 'close': 2}, {'mug': 13, 'cupboard': 3})

    def test_paraphrase_and_multiword_verb(self):
        self.assertEqual(self.mapper.map_text('picking up another mug'), (0, 13))
        self.assertEqual(self.mapper.map_text('place the mug in the cupboard'), (1, 13))

    def test_unknown_and_negation_are_not_positive_matches(self):
        self.assertEqual(self.mapper.map_text('do not close cupboard'), (None, None))
        self.assertEqual(self.mapper.map_text('place unknown-object'), (1, None))

    def test_snapshot_round_trip(self):
        restored = ActionMapper.from_dict(self.mapper.to_dict())
        self.assertEqual(restored.map_text('pick up mug'), (0, 13))

    def test_valid_verbs_are_not_lost_as_stopwords(self):
        mapper = ActionMapper({'start': 6, 'hand': 41, 'use': 76, 'take': 0},
                              {'dishwasher': 70, 'mug': 13, 'knife': 25})
        for text, gold in [('start dishwasher', (6, 70)), ('hand mug', (41, 13)),
                           ('use knife', (76, 25)), ('start taking mug', (0, 13))]:
            self.assertEqual(mapper.map_text(text), gold)

    def test_lexicon_ids_are_integers(self):
        for value in (True, 1.0, -1, float('nan')):
            with self.assertRaises(ValueError):
                ActionMapper({'take': value}, {'mug': 13})


if __name__ == '__main__':
    unittest.main()

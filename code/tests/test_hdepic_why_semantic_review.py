"""Aggregation checks: ambiguous cases stay visible, ordinal pairs are explicit."""
import unittest
from code.hdepic_why_semantic_review.report import metrics, paired

class SemanticReviewAggregation(unittest.TestCase):
    def rows(self):
        result=[]
        labels={'one':('incorrect','correct','partial'), 'two':('correct','partial','incorrect'),
                'three':('unjudgeable',)*3, 'four':('partial','partial','correct')}
        for rid,ll in labels.items():
            for model,label in zip(('base','epoch1','epoch2'),ll):
                result.append({'id':rid,'case':rid,'model':model,'label':label,
                    'unsupported_additional_claims':['extra'] if rid=='one' else [],
                    'visual_support':'not_established'})
        return result

    def test_ambiguous_sources_retained_in_denominator(self):
        x=metrics(self.rows())['epoch1']
        self.assertEqual(x['denominator_all'],4)
        self.assertEqual(x['judgeable'],3)
        self.assertEqual(x['correct_fraction_all'],.25)
        self.assertAlmostEqual(x['correct_fraction_judgeable'],1/3)
        self.assertEqual(x['label_counts']['unjudgeable'],1)
        self.assertEqual(x['unsupported_additional_claims'],1)

    def test_paired_regression_not_hidden_by_net_gain(self):
        x=paired(self.rows(),'base','epoch1')
        self.assertEqual(x['counts'],{'improvement':1,'regression':1,'tie':1,'unjudgeable_source':1})
        self.assertEqual(x['judgeable_pairs'],3)
        self.assertEqual(x['transition_matrix']['correct']['partial'],1)
        self.assertEqual(x['transition_matrix']['unjudgeable']['unjudgeable'],1)
        self.assertEqual(sum(sum(r.values()) for r in x['transition_matrix'].values()),4)

if __name__=='__main__':unittest.main()

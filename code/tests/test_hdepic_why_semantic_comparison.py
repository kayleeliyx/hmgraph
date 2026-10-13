import copy
import tempfile
import unittest
from pathlib import Path
from code.hdepic_why_semantic_comparison.report import metrics, paired, validate, verify_mapping
from code.hdepic_review_followup.fields import extract_field

class SemanticComparisonTests(unittest.TestCase):
    def record(self,m,rid,label,status='valid',fixed=False):
        return {'id':rid,'case':rid,'model':m,'label':label,'reference_status':status,'existing_fixed50':fixed,
                'unsupported_additional_claims':[],'visual_support':'not_established'}
    def test_denominators_do_not_drop_ambiguous_or_incorrect(self):
        rr=[self.record('new',str(i),x) for i,x in enumerate(('correct','partial','incorrect','unjudgeable'))]
        x=metrics(rr,('new',))['new']
        self.assertEqual(x['denominator_all'],4);self.assertEqual(x['judgeable'],3)
        self.assertEqual(x['correct_fraction_all'],.25);self.assertAlmostEqual(x['correct_fraction_judgeable'],1/3)
        self.assertEqual(sum(x['label_counts'].values()),4)
    def test_pairing_and_transition_direction(self):
        rr=[]
        for i,(a,b) in enumerate([('incorrect','correct'),('correct','partial'),('partial','partial'),('unjudgeable','unjudgeable')]):
            rr.extend([self.record('old',str(i),a),self.record('new',str(i),b)])
        p=paired(rr,'old','new')
        self.assertEqual(p['counts'],dict(improvement=1,regression=1,tie=1,unjudgeable_source=1))
        self.assertEqual(p['judgeable_pairs'],3)
        self.assertEqual(p['transition_matrix']['incorrect']['correct'],1)
    def test_missing_pair_fails_instead_of_dropping(self):
        with self.assertRaises(AssertionError):paired([self.record('old','x','correct')],'old','new')
    def test_duplicate_pair_fails(self):
        with self.assertRaises(AssertionError):paired([self.record('old','x','correct')]*2,'old','new')
    def test_mapping_bound_to_saved_raw_answers(self):
        texts={'qwen35':'<why>make room</why>','qwen38':'<why>clean the surface</why>'}
        answers={}
        for s,m in zip('AB',texts):
            f=extract_field(texts[m],'why')
            answers[s]={'raw_response':texts[m],'why':f['text'],'extraction_status':f['status'],'field_extraction':f}
        cases=[{'case':'S001','id':'x','answers':answers}]
        key={'S001':{'id':'x','slots':{'A':'qwen35','B':'qwen38'}}}
        predictions={m:{'x':{'prediction':t}} for m,t in texts.items()}
        verify_mapping(cases,key,predictions)
        key['S001']['slots']={'A':'qwen38','B':'qwen35'}
        with self.assertRaises(AssertionError):verify_mapping(cases,key,predictions)
    def fixture(self,root):
        (root/'page.jpg').write_bytes(b'fixture')
        c={'case':'S001','id':'a','frozen_source_decision':{'reference_status':'valid','purpose_core':'to reach the bowl'},
           'original_narration':'Lift it to reach the bowl.','answers':{'A':{'why':None},'B':{'why':'reach the bowl'}}}
        j={'case':'S001','id':'a','reviewer':'assistant fixture reviewer','reference_status':'valid','purpose_core':'to reach the bowl',
           'source_note':'The purpose is attached to lifting.', 'visual_observation':'The bowl is visible on the work surface.',
           'visual_reviewed':True,'inspection_pages':['page.jpg'],'answers':{s:{'label':l,'evidence_note':'Specific purpose is assessed against the source.',
           'unsupported_additional_claims':[],'visual_support':'not_established'} for s,l in [('A','incorrect'),('B','correct')]}}
        return c,j
    def test_failed_extraction_retained_as_incorrect(self):
        with tempfile.TemporaryDirectory() as d:
            c,j=self.fixture(Path(d));validate([c],{'S001':j},Path(d),1)
            j['answers']['A']['label']='partial'
            with self.assertRaises(AssertionError):validate([c],{'S001':j},Path(d),1)
    def test_source_policy_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            c,j=self.fixture(Path(d));j['purpose_core']='Lift it'
            with self.assertRaises(AssertionError):validate([c],{'S001':j},Path(d),1)
    def test_ambiguous_source_unjudgeable_for_every_answer(self):
        with tempfile.TemporaryDirectory() as d:
            c,j=self.fixture(Path(d));c['frozen_source_decision']['reference_status']=j['reference_status']='ambiguous_attachment'
            for a in j['answers'].values():a['label']='unjudgeable'
            validate([c],{'S001':j},Path(d),1)
            j['answers']['B']['label']='correct'
            with self.assertRaises(AssertionError):validate([c],{'S001':j},Path(d),1)
    def test_uninspected_images_cannot_have_visual_label(self):
        with tempfile.TemporaryDirectory() as d:
            c,j=self.fixture(Path(d));j['visual_reviewed']=False;j['inspection_pages']=[]
            with self.assertRaises(AssertionError):validate([c],{'S001':j},Path(d),1)
            for a in j['answers'].values():a['visual_support']='unreviewed'
            validate([c],{'S001':j},Path(d),1)

if __name__=='__main__':unittest.main()

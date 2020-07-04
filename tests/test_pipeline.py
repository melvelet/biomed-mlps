import unittest
from unittest.mock import MagicMock, patch

from pandas import DataFrame
from biomed.pipeline import Pipeline
from biomed.text_mining_manager import TextMiningManager
from biomed.properties_manager import PropertiesManager

class PipelineSpec( unittest.TestCase ):

    def test_it_is_a_Pipeline( self ):
        Pipe = Pipeline.Factory.getInstance( PropertiesManager(), MagicMock() )
        self.assertTrue( isinstance( Pipe, Pipeline ) )

    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_initializes_the_text_mining(
        self,
        TM: MagicMock,
    ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        Given = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )

        PM = PropertiesManager()
        PP = MagicMock()

        Pipe = Pipeline.Factory.getInstance( PM, PP )
        Pipe.pipe( Given )

        TM.assert_called_once_with(
            PM,
            PP
        )

    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_runs_the_text_miner_with_the_given_data( self, TM: MagicMock ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        Given = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )

        TMM = MagicMock( spec = TextMiningManager )
        TM.return_value = TMM

        Pipe = Pipeline.Factory.getInstance( PropertiesManager(), MagicMock() )
        Pipe.pipe( Given )

        TMM.setup_for_input_data.assert_called_once_with( Given )

    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_runs_the_text_miner_with_the_given_target_dimension( self, TM: MagicMock ):
        Given = "is_cancer"
        PM = PropertiesManager()
        PM.classifier = Given

        TMM = MagicMock( spec = TextMiningManager )
        TM.return_value = TMM

        Pipe = Pipeline.Factory.getInstance( PM, MagicMock() )
        Pipe.pipe( MagicMock() )

        TMM.setup_for_target_dimension.assert_called_once_with( Given )

    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_returns_the_computed_predictions( self, TM: MagicMock ):
        Expected = 42
        TMM = MagicMock( spec = TextMiningManager )
        TMM.get_binary_mlp_predictions.return_value = Expected
        TM.return_value = TMM

        Pipe = Pipeline.Factory.getInstance( MagicMock(), MagicMock() )
        self.assertEqual(
            Expected,
            Pipe.pipe( MagicMock() )
        )

    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_assigns_new_properties( self, _ ):
        PM = PropertiesManager()
        Expected = { "workers": 23, "classifier": PM.classifier }

        Pipe = Pipeline.Factory.getInstance( PM, MagicMock() )
        Pipe.pipe( MagicMock(), Expected )

        self.assertEqual(
            Expected[ "workers" ],
            PM.workers
        )

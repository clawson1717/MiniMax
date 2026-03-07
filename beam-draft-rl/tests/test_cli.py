import unittest
from unittest.mock import patch, MagicMock
from src.cli import main
import sys

class TestCLI(unittest.TestCase):
    def test_cli_help(self):
        with patch('argparse.ArgumentParser.print_help') as mock_help:
            with patch('sys.argv', ['cli.py']):
                main()
                mock_help.assert_called_once()

    def test_cli_train(self):
        with patch('src.cli.handle_train') as mock_handler:
             with patch('sys.argv', ['cli.py', 'train', '--model-name', 'gpt2', '--lr', '5e-5']):
                main()
                mock_handler.assert_called_once()
                args = mock_handler.call_args[0][0]
                self.assertEqual(args.model_name, 'gpt2')
                self.assertEqual(args.lr, 5e-5)

    def test_cli_evaluate(self):
        with patch('src.cli.handle_evaluate') as mock_handler:
             with patch('sys.argv', ['cli.py', 'evaluate', '--model-path', './models/peft_model']):
                main()
                mock_handler.assert_called_once()
                args = mock_handler.call_args[0][0]
                self.assertEqual(args.model_path, './models/peft_model')

    def test_cli_visualize(self):
        with patch('src.cli.handle_visualize') as mock_handler:
             with patch('sys.argv', ['cli.py', 'visualize-drafts']):
                main()
                mock_handler.assert_called_once()

    def test_cli_solve(self):
        with patch('src.cli.handle_solve') as mock_handler:
             with patch('sys.argv', ['cli.py', 'solve', 'Beam problem prompt', '--model-path', './models/peft_model']):
                main()
                mock_handler.assert_called_once()
                args = mock_handler.call_args[0][0]
                self.assertEqual(args.prompt, 'Beam problem prompt')
                self.assertEqual(args.model_path, './models/peft_model')

if __name__ == "__main__":
    unittest.main()

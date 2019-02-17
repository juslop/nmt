import unittest
#from unittest.mock import MagicMock, Mock, patch
from prepare import (
    process_line, post_process_line, handle_bracket, handle_quotes,
    read_dictionary
)

dictionary = {
    '<PAD>': 0,
    'kukkuu': 1,
    'abc': 2,
    '?': 3,
    '.': 4,
    '"': 5,
    '<EOL>': 6,
    '<UNK>': 7,
    '(': 8,
    ')': 9,
}
sequence_length = 6

class TestLineProcessing(unittest.TestCase):
    def test_lines(self):
        lines = [
            ('Kukkuu?', [1, 3, 6, 0, 0, 0]),
            ('Abc zaf', [2, 7, 6, 0, 0, 0]),
            ('"abc"', [5, 2, 5, 6, 0, 0]),
            ("'abc' --abc ****abc", [2, 2, 2, 6, 0, 0]),
            ('(abc)', [8, 2, 9, 6, 0, 0]),
            ('abc 123abc as23a 98u u986', [2, 6, 0, 0, 0, 0]),
        ]
        for s, l in lines:
            self.assertEqual(process_line(s, dictionary, 7, sequence_length), l)

    def test_brackets(self):
        sentence = "< a >< b > < z"
        processed = handle_bracket(sentence, ('<', '>'))
        self.assertEqual(processed, "<a><b> <z")
        sentence = '( a ) ( b ) ( c'
        processed = handle_bracket(sentence, ('(', ')'))
        self.assertEqual(processed, "(a) (b) (c")

    def test_quotes(self):
        sentence = '" haa " " huu "'
        processed = handle_quotes(sentence)
        self.assertEqual(processed, '"haa" "huu"')

    def test_post_process(self):
        sentence = "< abc > zaf , abc ."
        processed = post_process_line(sentence)
        self.assertEqual(processed, "<abc> zaf, abc.")
        sentence = '" haa " " huu " !'
        processed = post_process_line(sentence)
        self.assertEqual(processed, '"haa" "huu"!')
        sentence = 'abc !'
        processed = post_process_line(sentence)
        self.assertEqual(processed, 'Abc!')

    def test_dictionary(self):
        dictionary, word_to_index_map = read_dictionary(("finnish", "english"), "finnish-english")
        assert "is" in dictionary["english"].keys(), "is missing from dict"
        assert word_to_index_map["english"][dictionary["english"]["is"]] == "is", "indexes do not match"
        assert "mies" in dictionary["finnish"].keys(), "mies missing from dict"
        assert "nainen" in dictionary["finnish"].keys(), "nainen missing from dict"
        assert word_to_index_map["finnish"][dictionary["finnish"]["nainen"]] == "nainen", "indexes do not match"

if __name__ == '__main__':
    unittest.main()

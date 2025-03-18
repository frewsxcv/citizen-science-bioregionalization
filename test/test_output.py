import unittest
import os
import shutil
from src import output


class TestOutput(unittest.TestCase):
    def setUp(self):
        # Create a test output directory if it doesn't exist
        os.makedirs("test_output", exist_ok=True)
        # Save the original OUTPUT_DIR value
        self.original_output_dir = output.OUTPUT_DIR
        # Set the output directory to a test directory
        output.OUTPUT_DIR = "test_output"
    
    def tearDown(self):
        # Reset the OUTPUT_DIR value
        output.OUTPUT_DIR = self.original_output_dir
        # Remove the test output directory
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
    
    def test_ensure_output_dir(self):
        # Remove the test output directory if it exists
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
        
        # Call the ensure_output_dir function
        output.ensure_output_dir()
        
        # Check that the directory was created
        self.assertTrue(os.path.exists("test_output"))
        self.assertTrue(os.path.isdir("test_output"))
    
    def test_get_output_path(self):
        # Test that the function returns the correct path
        test_path = output.get_output_path("test.txt")
        self.assertEqual(test_path, os.path.join("test_output", "test.txt"))
    
    def test_normalize_path(self):
        # Test that paths without directory components are put in the output directory
        test_path = output.normalize_path("test.txt")
        self.assertEqual(test_path, os.path.join("test_output", "test.txt"))
        
        # Test that paths with directory components are not changed
        test_path = output.normalize_path("other_dir/test.txt")
        self.assertEqual(test_path, "other_dir/test.txt")
        
        # Test that paths already in the output directory are not changed
        test_path = output.normalize_path("test_output/test.txt")
        self.assertEqual(test_path, "test_output/test.txt")
    
    def test_get_geojson_path(self):
        # Test that the function returns the correct path
        test_path = output.get_geojson_path()
        self.assertEqual(test_path, os.path.join("test_output", output.GEOJSON_FILENAME))
    
    def test_get_html_path(self):
        # Test that the function returns the correct path
        test_path = output.get_html_path()
        self.assertEqual(test_path, os.path.join("test_output", output.HTML_FILENAME))
    
    def test_prepare_file_path(self):
        # Test that the function creates the directory if it doesn't exist
        test_path = os.path.join("test_output", "subdir", "test.txt")
        
        # Make sure the directory doesn't exist
        if os.path.exists(os.path.dirname(test_path)):
            shutil.rmtree(os.path.dirname(test_path))
        
        # Call the prepare_file_path function
        result_path = output.prepare_file_path(test_path)
        
        # Check that the directory was created and the path is correct
        self.assertTrue(os.path.exists(os.path.dirname(test_path)))
        self.assertTrue(os.path.isdir(os.path.dirname(test_path)))
        self.assertEqual(result_path, test_path)


if __name__ == "__main__":
    unittest.main() 
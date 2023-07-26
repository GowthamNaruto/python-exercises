import nbformat


def extract_python_code_from_notebook(file_path):
    with open(file_path, "r", encoding="utf-8") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    python_code = ""
    for cell in notebook_content.cells:
        if cell.cell_type == "code":
            python_code += cell.source + "\n\n"

    return python_code


def save_python_code_to_file(python_code, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(python_code)


# Replace "your_notebook.ipynb" with the path to your .ipynb file
notebook_file_path = "./Jupyter/Tensorflow.ipynb"
extracted_code = extract_python_code_from_notebook(notebook_file_path)

# Replace "output_file.py" with the desired output file name
output_file_path = "./Jupyter/output_file.py"
save_python_code_to_file(extracted_code, output_file_path)

print("Python code extracted and saved to:", output_file_path)

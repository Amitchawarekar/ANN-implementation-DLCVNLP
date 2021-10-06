with open("README.md", "r", encoding="utf-8") as f :
    long_description = f.read()

setup(
    name = "src",
    version = "0.0.1",
    author = "amitc"
    description = "A small package for ANN Implementation",
    LongDescription = long_description,
    LongDescription_content_type = "text/markdown",
    url = "https://github.com/Amitchawarekar/ANN-implementation-DLCVNLP",
    author_email = "amit.chawarekar@gmail.com",
    packages = ["src"],
    python_requires =">=3.7",
    install_requires =[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
    ]
)
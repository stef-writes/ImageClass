

### README.md

```markdown
# Image Classification and Captioning Project

This project performs image classification and captioning using two powerful models:
- **BLIP** (Bootstrapping Language-Image Pretraining) for generating captions.
- **CLIP** (Contrastive Language-Image Pretraining) for zero-shot image classification.

## Features

- **Image Captioning**: Generates detailed captions for images using the BLIP model.
- **Zero-shot Image Classification**: Matches images to provided text descriptions using the CLIP model.

## Project Structure

```
huggingface_image_analysis/
├── blip_clip_analysis.py    # Main script to run captioning and classification
├── config.json              # Configuration file for specifying images and prompts
├── images/                  # Directory containing input images
├── logs/                    # Directory where log files are stored
├── .gitignore               # Ignored files and directories
├── requirements.txt         # Python dependencies for the project
└── README.md                # This file
```

## Requirements

- **Python 3.9+**
- **Virtual environment**: (Optional but recommended) for managing project dependencies.

### Dependencies

Install the necessary dependencies listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/stef-writes/ImageClass.git
cd ImageClass
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your images

Place any images you want to analyze in the `images/` folder. Update the `config.json` file to reference these images.

## Running the Script

To run the captioning and classification:

```bash
python blip_clip_analysis.py
```

### Configuration (`config.json`)

```json
{
    "image_folder": "images/",
    "output_folder": "output/",
    "log_file": "logs/analysis_log.txt",
    "image_files": ["piano.jpeg", "waterfall.jpeg", "statue.jpg"],
    "text_prompts": ["a piano", "a waterfall", "a statue", "a dog", "a car"]
}
```

- **`image_folder`**: The folder where input images are stored.
- **`output_folder`**: Directory for storing outputs (e.g., logs).
- **`log_file`**: File for storing logs.
- **`image_files`**: List of image filenames to analyze.
- **`text_prompts`**: List of text prompts for CLIP to match with images.

## Example Output

```bash
Caption for piano.jpeg: a man playing a piano under a large tree
Best match for piano.jpeg: 'a piano' with confidence 0.2753

Caption for waterfall.jpeg: the yellowstone falls in yellowstone national park, wyoming
Best match for waterfall.jpeg: 'a waterfall' with confidence 0.2666

Caption for statue.jpg: a statue of a man holding a child
Best match for statue.jpg: 'a statue' with confidence 0.2726
```

## Contributing

Feel free to submit issues or pull requests if you'd like to contribute to the project.

## License

This project is licensed under the MIT License.
```

### Key Points:
- **Setup Instructions**: Explains how to set up the project with Git, virtual environments, and dependencies.
- **Running the Script**: Provides commands to run the analysis with `blip_clip_analysis.py`.
- **Configuration**: Describes the format and purpose of `config.json`.
- **Example Output**: Shows sample output of the script.

Once you’ve added this to your project, it will help others (and your future self) understand how to use it. Let me know if you’d like any adjustments!

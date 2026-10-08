# Medical Chatbot with LLMs-LangChain-Pinecone-Flask-AWS

# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/aishwarya4699/AI-Healthcare-Chatbot.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.12 -y
```

```bash
conda activate medibot
```


### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

OCR fallback also needs local system packages (not installed by pip):

```bash
# macOS
brew install tesseract poppler

# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr poppler-utils
```

### STEP 03- Train and evaluate the demo document classifier

The four-class page classifier (TF-IDF + Logistic Regression) is trained on **synthetic, non-PHI** texts in `classifier_data/`. That folder is separate from production ingest in `data/`.

```bash
python train_classifier.py
python evaluate_classifier.py
```

Evaluation prints accuracy, precision, recall, and F1, and writes `reports/classification_report.txt` plus `reports/confusion_matrix.png`.

These metrics only show that the demo pipeline runs. They are **not** representative of production clinical performance.

### STEP 04- Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# Place medical PDFs and images (png, jpg, jpeg, tif, tiff) in data/, then index
python store_index.py
```

`store_index.py` now runs a small Document AI step first: load files from `data/`, extract digital PDF text, OCR pages/images when native text is short, classify each page, then use the existing chunking, embeddings, and Pinecone RAG path.

OCR uses a prototype character-count heuristic (default 50, override with `OCR_MIN_EXTRACTED_CHARS`). In production this threshold should be tuned with evaluation data.

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone
- Tesseract OCR (scanned/image fallback)
- scikit-learn (TF-IDF + Logistic Regression document classification)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#With specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in AWS


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/<REPO_NAME>

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup Github secret keys:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - OPENAI_API_KEY

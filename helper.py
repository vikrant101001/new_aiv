import os
import config
#from PyPDF2 import PdfReader
from azure.storage.blob import BlobServiceClient


BLOB_CONNECTION_STRING = config.CONNECTION_STRING
BLOB_CONTAINER_NAME = config.CONTAINER_NAME



def document_crew(careteam_history, current_question):
    return "Document crew response"


def download_azure_blob(blob_path, blod_extension, user_id):
    print("Blob Path -", blob_path)
    blob_filename = os.path.basename(blob_path)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=BLOB_CONNECTION_STRING)

        container_client = blob_service_client.get_container_client(container=BLOB_CONTAINER_NAME)

        blob_download_file_path = f"download_{user_id}_{blob_filename}.{blod_extension}"

        try:
            with open(blob_download_file_path, "wb") as download_file:
                download_file.write(container_client.download_blob(blob_filename).readall())
                print(f"Blob downloaded to: {blob_download_file_path}")
        except Exception as e:
            print(f"Error downloading blob: {e}")
            return {"error": f"Error downloading blob: {e}"}, 500
                    
        return blob_download_file_path, 200
    except Exception as e:
        print(f"Error downloading blob: {e}")
        return {"error": f"Error downloading blob: {e}"}, 500



def extract_text_from_pdf(file_path):
    return "PDF Text"

def extract_text_from_image(file_path):
    # Extract text from the image file
    return "Image Text"

def extract_text_from_pdf(file_path):
    # Placeholder function for PDF extraction
    
    
    return f"Text extracted from PDF: {file_path}\n"

def extract_text_from_image(file_path):
    # Placeholder function for image extraction
    return f"Text extracted from image: {file_path}\n"
    

def text_extraction(metadata):
    user_id = "1234"
    description = ""
    for file_path in metadata:
        file_extension = file_path.split(".")[-1].lower() 
        if file_extension == "pdf":
            downloaded_pdf_file_path, status_code = download_azure_blob(metadata, file_extension, user_id)
            if status_code != 200:
                return downloaded_pdf_file_path
            else:
                extracted_text = extract_text_from_pdf(downloaded_pdf_file_path)
            
        elif file_extension in ["jpg", "jpeg", "png", "gif"]:
            downloaded_pdf_file_path, status_code = download_azure_blob(metadata, file_extension, user_id)
            if status_code != 200:
                return downloaded_pdf_file_path
            description += extract_text_from_image(file_path, user_id)
        else:
            print(f"Unsupported file type: {file_extension}")
    return description


#metadata = ["https://asif-test-bucket-practice.s3.ap-south-1.amazonaws.com/rust_programming_crab_sea.jpg", "path2.pdf"]
#description = text_extraction(metadata)
#print(description)

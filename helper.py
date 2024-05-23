import os

import ollama
import config
from azure.storage.blob import BlobServiceClient

import crews
import prompt_storage
from pypdf import PdfReader

from crews import pdf_description

BLOB_CONNECTION_STRING = config.CONNECTION_STRING
BLOB_CONTAINER_NAME = config.CONTAINER_NAME


def download_azure_blob(blob_path, blod_extension, user_id):
    print("Blob Path -", blob_path)
    blob_filename = os.path.basename(blob_path)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=BLOB_CONNECTION_STRING)

        container_client = blob_service_client.get_container_client(container=BLOB_CONTAINER_NAME)

        blob_download_file_path = f"download_{user_id}_{blob_filename}"

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


def llava_image_extraction(image_list: list[str]) -> ollama.chat:
    try:
        prompt = prompt_storage.prompt_for_llava()
        res = ollama.chat(
            model="llava",
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': image_list,
                }
            ]
        )
        response = res['message']['content']
        print("Response from LLAVA: \n\n", response)
        return response, 200
    except Exception as e:
        print(f"Error: {e}")
        return {"Error from LLAVA": f"Error: {e}"}, 500


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)

    # Print the number of pages in the PDF file
    num_pages = len(reader.pages)
    print(f"Number of pages: {num_pages}")
    final_extracted = ""
    # Iterate over all the pages and extract text from each one
    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text()
        print(f"Page {i + 1}:\n{text}\n")
        text_on_page = f"Page {i + 1}:\n{text}\n"
        final_extracted += text_on_page

    print(f"Text extracted from PDF: {file_path}\n and Text: {final_extracted}")

    return final_extracted, 200


def download_and_check(file_path, file_extension, user_id):
    try:
        downloaded_file_path, status_code = download_azure_blob(file_path, file_extension, user_id)
        if status_code != 200:
            return downloaded_file_path, 400
        return downloaded_file_path, 200
    except Exception as e:
        print(f"Error during download: {e}")
        return str(e), 400


def handle_pdf(file_path):
    try:
        extracted_text, status_code = extract_text_from_pdf(file_path)
        if status_code != 200:
            return extracted_text, 400
        pd_description, status_code = crews.pdf_description(extracted_text)
        if status_code != 200:
            return pd_description, 400
        return pd_description, 200
    except Exception as e:
        print(f"Error during PDF handling: {e}")
        return str(e), 400
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {file_path}, {e}")


def handle_image(file_name):
    try:
        current_directory = os.getcwd()
        print(current_directory)
        file_path = os.path.join(current_directory, file_name)
        print("The full path of the file is:", file_path)
        file_path = [str(file_path)]
        llava_extraction_response, status_code = llava_image_extraction(file_path)
        if status_code != 200:
            return llava_extraction_response, 400
        return llava_extraction_response, 200
    except Exception as e:
        print(f"Error during image handling: {e}")
        return str(e), 400
    # finally:
    #     try:
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #             print(f"Deleted file: {file_path}")
    #     except Exception as e:
    #         print(f"Error deleting file: {file_path}, {e}")


def text_extraction(metadata, user_id):
    description = ""
    try:
        for file_path in metadata:
            file_extension = file_path.split(".")[-1].lower()

            if file_extension == "pdf":
                downloaded_file_path, status_code = download_and_check(file_path, file_extension, user_id)
                if status_code != 200:
                    return downloaded_file_path, 400
                pd_description, status_code = handle_pdf(downloaded_file_path)
                if status_code != 200:
                    return pd_description, 400
                description += pd_description

            elif file_extension in ["jpg", "jpeg", "png", "gif"]:
                downloaded_file_path, status_code = download_and_check(file_path, file_extension, user_id)
                if status_code != 200:
                    return downloaded_file_path, 400
                llava_extraction_response, status_code = handle_image(downloaded_file_path)
                if status_code != 200:
                    return llava_extraction_response, 400
                description += llava_extraction_response

            else:
                print(f"Unsupported file type: {file_extension}")
                return f"Unsupported file type: {file_extension}", 400

        return description, 200
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return str(e), 400

#metadata = ["https://asif-test-bucket-practice.s3.ap-south-1.amazonaws.com/rust_programming_crab_sea.jpg", "path2.pdf"]
#description = text_extraction(metadata)
#print(description)

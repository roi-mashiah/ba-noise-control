import json
from PIL import Image
import requests
from io import BytesIO
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor as executor

API_KEY = "YOUR_API_KEY"
SEARCH_ENGINE_ID = "YOUR_CX_ID"


class GoogleSearchClient:
    def __init__(self, api_key, cx):
        self.api_key = api_key
        self.cx = cx
        try:
            self.service = build("customsearch", "v1", developerKey=self.api_key)
        except Exception as ex:
            print(f"Problem instantiating the Google Search Client!\n{ex}")
            self.service = None

    def search_images(self, query, num_results=5):
        if self.service is None:
            print("No Google Search service. Exiting search function...")
            return []
        res = (
            self.service.cse()
            .list(q=query, cx=self.cx, searchType="image", num=num_results)
            .execute()
        )

        image_urls = [item["link"] for item in res["items"]]
        return image_urls

    @staticmethod
    def get_image(image_urls):
        for image_url in image_urls:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGBA")
                    return img
                else:
                    raise PermissionError(
                        f"URL: {image_url}\nResponse Code: {response.status_code}\nCan't download image!"
                    )
            except Exception as ex:
                print(ex)
                continue

    def search_and_fetch_image(self, query):
        urls = self.search_images(query)
        return self.get_image(urls)


def get_new_dimensions(box, grid_size=(512, 512)):
    x_min = int(box[0] * grid_size[1])
    y_min = int(box[1] * grid_size[0])
    x_max = int(box[2] * grid_size[1])
    y_max = int(box[3] * grid_size[0])
    new_size = (x_max - x_min, y_max - y_min)
    position = (x_min, y_min)
    return new_size, position


def create_subject_query(input_obj, subject_tokens):
    tokens = input_obj["prompt"].split(" ")
    subject_query = " ".join([tokens[s - 1] for s in subject_tokens])
    subject_query = f"one {subject_query} with transparent background in png format"
    return subject_query


def run(input_obj):
    try:
        # query for background
        background = client.search_and_fetch_image(f"{input_obj['background']}").resize(
            grid_size
        )
        for subject_tokens, box in zip(input_obj["references"], input_obj["boxes"]):
            # query for subject
            subject_query = create_subject_query(input_obj, subject_tokens)
            subject_orig = client.search_and_fetch_image(subject_query)
            # paste on background
            new_size, position = get_new_dimensions(box)
            patch = subject_orig.resize(new_size)
            background.paste(patch, position, mask=patch)

        return background.convert("RGB")
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    client = GoogleSearchClient(API_KEY, SEARCH_ENGINE_ID)
    grid_size = (512, 512)

    with open("db_inputs.json", "r") as reader:
        inputs = json.load(reader)

    with executor(max_workers=1) as exec:
        m = exec.map(run, list(inputs.items()))

    res = [i for i in m]

import os
import httpx
import random
import asyncio
import argparse


async def send_request(client, image_path, idx, api_url):
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        files = {"file": ("image.jpg", image_data, "image/jpeg")}
        response = await client.post(api_url, files=files)
        print(f"Request {idx}: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"Request {idx} failed with exception: {e}")


async def main(args):
    image_files = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in the folder: {args.image_folder}")
        return

    async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
        tasks = [
            send_request(client, random.choice(image_files), idx, args.api_url)
            for idx in range(args.num_requests)
        ]
        await asyncio.gather(*tasks)

# Entry point
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Send concurrent HTTP requests with random images from a folder.")
    parser.add_argument(
        "--num_requests", 
        type=int, 
        default=100, 
        help="Number of requests to send (default: 100)"
    )
    parser.add_argument(
        "--api_url", 
        type=str, 
        default="http://0.0.0.0:8000/predict/", 
        help="Destination API URL (default: http://0.0.0.0:8000/predict/)"
    )
    parser.add_argument(
        "--image_folder", 
        type=str, 
        default="images", 
        help="Path to the folder containing image files (default: 'images')"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main(args))

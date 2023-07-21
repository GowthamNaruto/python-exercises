# pip install captcha
# pip install pillow
from captcha.image import ImageCaptcha
from PIL import Image
import os

# Define the generate_captcha_text and the length of the captcha


def generate_captcha_text(length=6):
    import string
    import random
    # Return the captcha text with random ascii and digits combined
    # How many ascii and digits, will be randomly generated
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# Define generat_and_save_captcha function
# And give image width, height and folder path
# And give image width, height and folder path
def generate_and_save_captcha(image_width=300, image_height=100,
                              captcha_length=6, save_folder='./captchas'):
    image = ImageCaptcha(width=image_width, height=image_height)
    captcha_text = generate_captcha_text(captcha_length)
    save_path = os.path.join(save_folder, 'CAPTCHA.png')
    data = image.generate(captcha_text)
    image.write(captcha_text, save_path)
    return captcha_text


if __name__ == '__main__':
    save_folder_path = './captchas'
    captcha_text = generate_and_save_captcha(save_folder=save_folder_path)
    print("CAPTCHA text:", captcha_text)
captcha_image = Image.open('./captchas/CAPTCHA.png')
captcha_image.show()

from PIL import Image
import glob, os

size = 224, 224

for file in glob.glob("C:/! random/ML Proj/dataset/*/*/*.jpeg"):
    filename, ext = os.path.splitext(file)
    with Image.open(file) as img:
        img.thumbnail(size)
        final_img = Image.new('RGB', size) # creates black background for padding
        x = (224 - img.width) // 2
        y = (224 - img.height) // 2
        final_img.paste(img, (x, y))
        final_img.save(filename + '_resized.jpeg', 'JPEG')
        os.remove(file)
    print(f'{file} has been resized and the original removed')

print('script complete')
import glob, os
# For use in removing imperfect files I created during preprocessing
for file in glob.glob("C:/! random/ML Proj/dataset/*/*/*_resized.jpeg"):
    os.remove(file)
    print(f"deleted {file}")
print("cleanup done")
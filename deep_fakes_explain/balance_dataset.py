import os
import random

real_path = r"dataset/test_set/Real"
fake_path = r"dataset/test_set/Fake"

# get image lists
real_images = os.listdir(real_path)
fake_images = os.listdir(fake_path)

real_count = len(real_images)
fake_count = len(fake_images)

print("Real images:", real_count)
print("Fake images:", fake_count)

# if fake has more images
if fake_count > real_count:

    diff = fake_count - real_count
    print("Deleting", diff, "images from Fake")

    delete_list = random.sample(fake_images, diff)

    for img in delete_list:
        os.remove(os.path.join(fake_path, img))

    print("Dataset balanced!")

else:
    print("Fake images are already balanced or fewer.")
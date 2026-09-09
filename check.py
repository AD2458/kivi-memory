from doublemetaphone import doublemetaphone

names = [
    "Aditya", "Aaditya",
    "kobi", "kivi",
    "Deepak", "Dipak",
    "Pooja", "Puja",
    "Shweta", "Sweta", "Shwetha",
    "Karthik", "Kartik", "Karthick",
    "Lakshmi", "Laxmi",
    "Srinivas", "Shrinivas",
    "Vivek", "Viveck",
    "Raghav", "Ragav",
    "Pranav", "Pranab",
    "Suresh", "Sures",
    "satriya", "Kshatriya",
]

for name in names:
    primary, secondary = doublemetaphone(name)
    print(f"{name:12} → Primary: {primary:8} Secondary: {secondary}")
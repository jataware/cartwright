from cartwright.categories import geos, dates,partial_dates, misc, timespans
import importlib
from inspect import isclass

# def return_all_categories_classes():
#     all_categories=[]
#     classes_geo = [x for x in dir(geos) if isclass(getattr(geos, x))]
#     classes_dates = [x for x in dir(dates) if isclass(getattr(dates, x))]
#     classes_partial_dates = [x for x in dir(partial_dates) if isclass(getattr(partial_dates, x))]
#     classes_misc = [x for x in dir(misc) if isclass(getattr(misc, x))]
#     all_classes=list(set(classes_misc + classes_dates + classes_partial_dates + classes_geo))
#     print(all_classes)
#     for name in all_classes:
#         if name in ["Fake","Faker"]:
#             pass
#         else:
#             all_categories.append(str(name))
#     return all_categories


# TODO: this should be dynamically generated based on e.g. files in the categories directory
def return_all_category_classes_and_labels():
    all_classes={}
    dirs={"geos":geos,"dates":dates,"timespans":timespans,"misc":misc, "partial_dates":partial_dates}
    classes_all={}
    for dir_ in dirs:
        classes_all[dir_]=[x for x in dir(dirs[dir_]) if isclass(getattr(dirs[dir_], x))]

    for dir_ in dirs:

        module = importlib.import_module(f"cartwright.categories.{dir_}")
        for class_name in classes_all.get(dir_):
            if class_name in ["Fake", "Faker", "DateBase","GeoBase", "CategoryBase", "defaultdict"]:
                pass
            else:
                try:
                    class_ = getattr(module, class_name)
                    instance = class_()
                    all_classes[instance.return_label()]=instance
                except:
                    print(class_name)

    return all_classes

from collections import defaultdict

def generate_label_id(all_labels):

    label2id = {}
    count = 0
    for label in all_labels:
        label2id[label] = count
        count += 1

    return label2id
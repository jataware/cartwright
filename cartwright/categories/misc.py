from cartwright.categories.CategoryBases import CategoryBase
import numpy as np
import logging
from cartwright.utils import build_return_standard_object, fuzzy_match

class first_name(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()


class percent(CategoryBase):
    def __init__(self):
        super().__init__()
        self.percent=[
            100,
            1000,
            10000
        ]

    def generate_training_data(self):
        scale = lambda x: x / np.random.choice(self.percent)
        val = str(np.random.choice(scale(np.array(list(range(0, 100))))))
        return self.class_name(), str(val)

    def validate(self,values):
        return self.not_classified()

class ssn(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()

class language_name(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()

class country_name(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, "country")())

    def validate(self,values):
        return self.not_classified()

class phone_number(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()


class zipcode(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()


class paragraph(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()

class pyfloat(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()

class email(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()


class prefix(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()

class pystr(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.not_classified()


class boolean(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self, values):
        try:
            logging.info("Start boolean validation ...")
            bool_arr = ["true", "false", "T", "F"]
            bool_array = []
            for bools in values:
                for b in bool_arr:
                    try:
                        bool_array.append(fuzzy_match(bools, b, ratio=85))
                    except Exception as e:
                        logging.error(f"bool_f - {values}: {e}")

            if np.count_nonzero(bool_array) >= (len(values) * 0.85):

                return build_return_standard_object(category='boolean', subcategory=None, match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)
        except Exception as e:
            logging.error(f'Boolean validation: {e}')
            return build_return_standard_object(category=None, subcategory=None, match_type=None)


class boolean_letter(CategoryBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), np.random.choice(['t', 'f','T', 'F'])

    def validate(self, values):
            try:
                logging.info('Start boolean validation ...')
                bool_arr = ['t', 'f', 'T', 'F']
                bool_array = []
                for bools in values:
                    for b in bool_arr:
                        try:
                            bool_array.append(fuzzy_match(bools, b, ratio=98))
                        except Exception as e:
                            logging.error(f"bool_letter_f -{values}: {e}")

                if np.count_nonzero(bool_array) >= (len(values) * .85):

                    return build_return_standard_object(category='boolean', subcategory=None, match_type='LSTM')
                else:
                    return build_return_standard_object(category=None, subcategory=None, match_type=None)
            except Exception as e:
                logging.error(f'boolean_letter error: {e}')
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

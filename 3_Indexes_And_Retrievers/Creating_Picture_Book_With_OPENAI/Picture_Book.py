# With this, we can send the chat model a page from our book, the function, and 
# instructions to infer the details from the provided page. In return, we get structured 
# data that we can use to form a great Stable Diffusion prompt!

import json
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.schema import (
  HumanMessage,
  SystemMessage
)

get_visual_description_function = [{
    'name': 'get_passage_setting',
    'description': 'Generate and describe the visuals of a passage in a book. Visuals only, no characters, plot, or people.',
    'parameters': {
        'type': 'object',
        'properties': {
            'setting': {
                'type': 'string',
                'description': 'The visual setting of the passage, e.g. a green forest in the pacific northwest',
            },
            'time_of_day': {
                'type': 'string',
                'description': 'The time of day of the passage, e.g. nighttime, daytime. If unknown, leave blank.',
            },
            'weather': {
                'type': 'string',
                'description': 'The weather of the passage, eg. rain. If unknown, leave blank.',
            },
            'key_elements': {
                'type': 'string',
                'description': 'The key visual elements of the passage, eg tall trees',
            },
            'specific_details': {
                'type': 'string',
                'description': 'The specific visual details of the passage, eg moonlight',
            }
        },
        'required': ['setting', 'time_of_day', 'weather', 'key_elements', 'specific_details']
    }
}]


response= self.chat([HumanMessage(content=f'{page}')],functions=get_visual_description_function)


function_dict = json.loads(response.additional_kwargs['function_call']['arguments'])


setting = function_dict['setting']

def get_pages(self):
        pages = self.chat([HumanMessage(content=f'{self.book_text_prompt} Topic: {self.input_text}')]).content
        return pages

def get_prompts(self):
    base_atmosphere = self.chat([HumanMessage(content=f'Generate a visual description of the overall lightning/atmosphere of this book using the function.'
                                                          f'{self.book_text}')], functions=get_lighting_and_atmosphere_function)
    summary = self.chat([HumanMessage(content=f'Generate a concise summary of the setting and visual details of the book')]).content

def generate_prompt(page, base_dict):
     prompt = self.chat([HumanMessage(content=f'General book info: {base_dict}. Passage: {page}. Infer details about passage if they are missing, '
                                                     f'use function with inferred detailsm as if you were illustrating the passage.')],
    functions=get_visual_description_function)
     
 for i, prompt in enumerate(prompt_list):
        entry = f"{prompt['setting']}, {prompt['time_of_day']}, {prompt['weather']}, {prompt['key_elements']}, {prompt['specific_details']}, " \
                f"{base_dict['lighting']}, {base_dict['mood']}, {base_dict['color_palette']}, in the style of {style}"



import deeplake

class SaveToDeepLake:
    def __init__(self, buildbook_instance, name=None, dataset_path=None):
        self.dataset_path = dataset_path
        try:
            self.ds = deeplake.load(dataset_path, read_only=False)
            self.loaded = True
        except:
            self.ds = deeplake.empty(dataset_path)
            self.loaded = False

        self.prompt_list = buildbook_instance.sd_prompts_list
        self.images = buildbook_instance.source_files

    def fill_dataset(self):
        if not self.loaded:
            self.ds.create_tensor('prompts', htype='text')
            self.ds.create_tensor('images', htype='image', sample_compression='png')
        for i, prompt in enumerate(self.prompt_list):
            self.ds.append({'prompts': prompt, 'images': deeplake.read(self.images[i])})




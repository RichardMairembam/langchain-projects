from dotenv import load_dotenv
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.6)
def generate_restaurant_name_and_menu(cuisine):
    
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template= "I want to open a restaurant for {cuisine} food. Suggest a fency name for this.")
    prompt_template_name.format(cuisine='Indian')
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template= "I want to open a restaurant for {cuisine} food. Suggest a fency name for this.")
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_menu = PromptTemplate(
        input_variables=['restaurant_name'],
        template= "Suggest some menu items for {restaurant_name}. Return it as a comma separated list.")
    menu_chain = LLMChain(llm=llm, prompt=prompt_template_menu, output_key="menu_items")
    chain = SequentialChain(
        chains=[name_chain,menu_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name','menu_items']
    )
    response = chain.invoke({'cuisine':cuisine})
    
    return response

if __name__ == '__main__':
     print(generate_restaurant_name_and_menu("Indian"))
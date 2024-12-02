
import os

# Specify the GPU number (e.g., 0 for the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
from unsloth import FastLanguageModel
# Check if CUDA is available and which device is being used
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.current_device()}")
else:
    print("CUDA is not available.")



######### 0. prepare the prompt message data #########
import pandas as pd

# prompting tutorial https://llama.meta.com/docs/how-to-guides/prompting
info_persona = """you are a native Chinese speaker and social science annotator, 
your task is to label the moral foundation values expressed in the given Chinese documents.
"""

info_mft = """moral foundation values are the core values that underlie moral reasoning from the moral foundation theory.
the five moral foundations are: care, fairness, loyalty, authority, and sanctity. And they refer to the following moral intuitions, each includes both vice/virtue pairs:
- care: related to our long evolution as mammals with attachment systems and an ability to feel (and dislike) the pain of others. It underlies the virtues of kindness, gentleness, and nurturance.
- fairness: related to the evolutionary process of reciprocal altruism. It underlies the virtues of justice and rights. 
- loyalty:  our long history as tribal creatures able to form shifting coalitions. It is active anytime people feel that its “one for all and all for one.” It underlies the virtues of patriotism and self-sacrifice for the group. 
- authority: This foundation was shaped by our long primate history of hierarchical social interactions. It underlies virtues of leadership and followership, including deference to prestigious authority figures and respect for traditions.
- sanctity: This foundation was shaped by the psychology of disgust and contamination. It underlies notions of striving to live in an elevated, less carnal, more noble, and more “natural” way (often present in religious narratives). This foundation underlies the widespread idea that the body is a temple that can be desecrated by immoral activities and contaminants (an idea not unique to religious traditions). It underlies the virtues of self-discipline, self-improvement, naturalness, and spirituality. 
"""

info_instruct = """you should follow the given principles to label the moral foundation values in the give documents:
1. identify the moral foundation value only from the 5 given ones.
2. if the document expresses more than 1 foundation value, label all prominent values, but in total should be less than 3 values.
3. provide a brief rationale for the each labelling, which should be less than 20 tokens. 
4. labels the value in english, 
5. rationales should be in the same lanaguage as the document
6. if the document does not express any of the 5 values, label it as 'none' and provide a brief rationale.
7. if the document can not be labelled into any of the 5 values, label it as 'unknown' and provide a brief rationale.
8. consider the Chinese cultural context of the document when labelling the values.
"""

info_example = """Here are some examples of the expected inputs and outputs:
- input: {user: '为了厘清1978年以后的思想混乱状况，中共中央认为，我们必须坚持社会主义道路；必须坚持无产阶级专政；必须坚持共产党的领导；必须坚持马列主义、毛泽东思想。这一定调为全党全国指明了方向。此次事件再次证明，一小撮卖国集团绑架民意，危害地区稳定。我们抛头颅洒热血也要和卖国贼斗争到底，卖国集团不除，国家永无宁日。对待卖国汉奸，我们应该只有炮弹没有糖衣。'}
- output: {gpt: 'authority, loyalty'}

- input: {user: '今年四月有上千名储户合计超过12亿的存款无法取出，这些储户有一个共同点，他们都是在相同的5家银行办理了储蓄业务。新闻爆出后，自然也让相关的其他储户心里亮起了警灯，结果所有人都无法正常取款，很多储户不得已便去银行门口聚集反抗，以此来维护他们的权益。最后，经统计有397亿存款不翼而飞，这些储户仍在用各种方式上访、投诉，以追回自己的存款。'}
- output: {gpt: 'authority, fairness'}

- input: {user: '妈祖，中国古代神话中的海神，又称天后、天上圣母等等 ，是历代船工、海员、旅客、商人和渔民共同信奉的神祇，祈求保佑顺风和安全。20世纪80年代，联合国有关机构授予妈祖“和平女神”称号。目前，全世界共有上万座从湄洲祖庙分灵的妈祖庙，有3亿多人信仰妈祖。各妈祖庙都会举办各式各样的纪念活动。'}
- output: {gpt: : 'sanctity'}
"""

###### Chinese version prompts ######
# info_persona = """
# 你是一名以中文为母语的社会科学研究员，
# 你的任务是对给定的中文文档中反应的道德基础价值进行标注。
# """

# info_mft = """
# 道德基础价值是道德基础理论中解释道德推理的核心价值观。
# 五大道德基础包括：关怀、公平、忠诚、权威和神圣。它们分别对应以下道德直觉，每个基础道德都包括正反两个方面：
# - 关怀：与我们作为哺乳动物长期进化的依恋系统及感知他人痛苦的能力有关。它支撑着善良、温和和养育等美德。
# - 公平：与互惠利他主义的进化过程有关。它支撑着正义和权利的美德。
# - 忠诚：源自我们作为部落生物能够组成变动联盟的悠久历史。它支撑着爱国主义和为群体自我牺牲的美德。
# - 权威：由我们等级社会互动历史塑造的。它支撑着领导和追随的美德，包括对有声望的权威人物的尊敬和对传统的尊重。
# - 神圣：由厌恶感的心理塑造。它支撑着追求更高尚、超越世俗、更“自然”生活的观念（常见于宗教叙事中）。这一基础支撑着身体是神圣不可侵犯的思想，即身体可能因不道德的行为和污染而被玷污（这一想法并不限于宗教传统）。它支撑着自律、自我提升、自然性和精神性的美德。
# """

# info_instruct = """
# 你应遵循以下原则对文档中的道德基础价值进行标注：
# 1.	仅从提供的5个道德基础中识别价值。
# 2.	如果文档表达了多于一个道德基础价值，标注所有显著的价值，但总数不得超过3个。
# 3.	每个标注都应提供简短的中文理由，理由字数不得超过20个字。
# 4.	在输出结果中使用英文标注道德基础价值，
# 5.	如果文档未表达任何5个基础中的价值，标注为’none’并提供简短的理由。
# 6.	如果文档无法归类为任何一个基础，标注为’unknown’并提供简短的理由。
# 7.	在标注价值时应考虑文档的中国文化背景。
# """

# info_output = """
# 输出应为以下json格式，每个文档的标注用大阔号括起来，并用换行符分隔：
# {id: '复制文档的id', value: '道德基础价值', rationale: '简短的价值理由'}/n
# """

# info_example = """
# 以下是3个示例：
# - input: {id: '0', text: '为了厘清1978年以后的思想混乱状况，中共中央认为，我们必须坚持社会主义道路；必须坚持无产阶级专政；必须坚持共产党的领导；必须坚持马列主义、毛泽东思想。这一定调为全党全国指明了方向。此次事件再次证明，一小撮卖国集团绑架民意，危害地区稳定。我们抛头颅洒热血也要和卖国贼斗争到底，卖国集团不除，国家永无宁日。对待卖国汉奸，我们应该只有炮弹没有糖衣。'}
# - output: {id: '0', value: 'authority, loyalty', rationale: '该文档描述了中共中央对于社会主义道路的坚持，这体现了对于领导权威的尊重;文中提到了卖国集团绑架民意，危害地区稳定，这体现了对于国家的忠诚。'}\n

# - input: {id: '1', text: '今年四月有上千名储户合计超过12亿的存款无法取出，这些储户有一个共同点，他们都是在相同的5家银行办理了储蓄业务。新闻爆出后，自然也让相关的其他储户心里亮起了警灯，结果所有人都无法正常取款，很多储户不得已便去银行门口聚集反抗，以此来维护他们的权益。最后，经统计有397亿存款不翼而飞，这些储户仍在用各种方式上访、投诉，以追回自己的存款。'}
# - output: {id: '1', value: 'authority, fairness', rationale: '该文档描述了储户在银行存款无法取出的事件，这种情况违反了储户的权益和公平，表现了公平性;储户们聚集反抗，体现了反抗精神和对权威的质疑。'}\n

# - input: {id: '2', text: '妈祖，中国古代神话中的海神，又称天后、天上圣母等等 ，是历代船工、海员、旅客、商人和渔民共同信奉的神祇，祈求保佑顺风和安全。20世纪80年代，联合国有关机构授予妈祖“和平女神”称号。目前，全世界共有上万座从湄洲祖庙分灵的妈祖庙，有3亿多人信仰妈祖。各妈祖庙都会举办各式各样的纪念活动。'}
# - output: {id: '2', value: 'sanctity', rationale: '该文档描述了妈祖的神圣性，妈祖是中国古代神话中的海神，是历代船工、海员、旅客、商人和渔民共同信仰的神明，祈求保佑顺风和安全。'}\n
# """


######## 1. wrap the training and inference in the loop #####
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import re
import csv
from tqdm import tqdm


# # 5. load the model and run the inference

system_message = info_persona + info_mft + info_instruct + info_example
max_seq_length = 4028 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ft_prompt_ch_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

###### model inference
# load the benchmark data
benchmark = pd.read_csv("/data/scro4316/thesis/paper3/benchmarkset.csv")
# add the column index as the id (to track the output)
# benchmark["id"] = benchmark.index.astype(str)

inference_list = []
# create a list, each has three dictionaries, each has two keys: role and content
for i in range(0, len(benchmark)):
    inference_single = [
        {"from": "system", "value": system_message},
        {"from": "user", "value": benchmark.loc[i, 'text']},
        # {"from": "gpt", "value": dataset_ft.loc[i, 'labels']}
    ]
    # prompt_single_dict = {"conversations": prompt_single}
    inference_list.append(inference_single)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

def generate_input(message): 
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        padding=True, 
        truncation=True,
    ).to("cuda")
    return inputs

# define a function to extract the user and assistant content from the generated text
def extract_user_assistant(generated_text):
    # Define regex patterns for user and assistant
    user_pattern = r"user\s*(.*?)\s*assistant"
    assistant_pattern = r"assistant\s(.*)"

    # Extract user content
    user_match = re.search(user_pattern, generated_text, re.DOTALL)
    user_content = user_match.group(1).strip() if user_match else None

    # Extract assistant content
    assistant_match = re.search(assistant_pattern, generated_text, re.DOTALL)
    assistant_content = assistant_match.group(1).strip() if assistant_match else None

    return {"user": user_content, "assistant": assistant_content}


for i in tqdm(range(0, len(inference_list)), desc="Processing Inferences"):
    message = inference_list[i]
    inputs = generate_input(message)
    outputs = model.generate(input_ids = inputs, 
                        max_new_tokens = 512, 
                        use_cache = True,
                        temperature = 0.1,)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    extracted_values = extract_user_assistant(generated_text)

    # Define the CSV file path
    csv_file_path = f"ft_prompt_ch_model.csv"
    
    ## Check if the file exists before writing
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode without writing a header
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=extracted_values.keys())
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()
        writer.writerow(extracted_values)
    
    print(f"\{i} records have been successfully inferred and saved to the csv.")

# # Local saving - update the batch model and tokenizer, and save them with the same path
# model_save_path = "llama_ft_enandzh_prompt_en_model_batch"
# model.save_pretrained(model_save_path) 
# tokenizer.save_pretrained(model_save_path)
# print(f"Model and tokenizer successfully overwritten to {model_save_path}.")


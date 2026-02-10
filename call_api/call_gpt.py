import os
import asyncio
from openai import AsyncOpenAI
from datasets import load_dataset

client = AsyncOpenAI(
    api_key="Your API Key"
)


MAX_RETRIES = 3


async def call_gpt_async(input_prompt, num_samples=1, temperature=1.0, top_p=1.0, retries=0):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful translator and only output the result."},
                {"role": "user", "content": input_prompt},
            ],
            max_tokens=256,
            temperature=temperature,
            top_p=top_p,
            n=num_samples,
            stop=["\n", "\t"],
        )
        outputs = []
        for choice in response.choices:
            text = choice.message.content
            text = text.strip().split('\n')[0]
            outputs.append(text)
        if len(outputs) == num_samples:
            return outputs
        else:
            raise ValueError("Mismatch between the number of outputs requested and received.")
    except Exception as e:
        print(f"Error: {e}")
        if retries < MAX_RETRIES:
            print(f"Retrying... Attempt {retries + 1}")
            await asyncio.sleep(5)
            return await call_gpt_async(input_prompt, num_samples, temperature, retries + 1)
        else:
            if len(outputs) > num_samples:
                return outputs[:num_samples]
            else:
                return outputs + ["."] * (num_samples - len(outputs))  # in case empty output


async def process_sample(lang_pair, src_sent, tgt_sent, num_samples, temperature, top_p):
    lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German",
                 "cs": "Czech", "ru": "Russian", "is": "Icelandic", "es": "Spanish",
                 "uk": "Ukrainian", "ja": "Japanese", "hi": "Hindi", "pt": "Portuguese",
                 "fr": "French", "it": "Italian", "ko": "Korean", "nl": "Dutch"}
    s, t = lang_pair.split('-')
    src_lang, tgt_lang = lang_name[s], lang_name[t]
    if t == "en":
        src_sent, tgt_sent = tgt_sent, src_sent
    input_prompt = f"### Translate this sentence from {src_lang} to {tgt_lang}, {src_lang}: {src_sent}\n### {tgt_lang}:"

    outputs = await call_gpt_async(input_prompt, num_samples, temperature, top_p)

    return src_sent, tgt_sent, outputs


async def main():
    lang_pairs = ["en-es", "en-ru", "en-zh", "en-fr", "en-nl", "en-it", "en-pt", "en-ko"]


    num_samples = 1
    temperature = 1.0
    data_list = ["wmt24_testset"]
    model_name = "gpt-4o-mini"
    top_p = 0.98

    for data in data_list:
        for lang_pair in lang_pairs:
            src, tgt = lang_pair.split('-')

            src_file = os.path.join("../src/llama_recipes/customer_data/{}/test".format(data), "{}-{}".format(src, tgt),
                                    "test.{}-{}.{}".format(src, tgt, src))
            tgt_file = os.path.join("../src/llama_recipes/customer_data/{}/test".format(data), "{}-{}".format(src, tgt),
                                    "test.{}-{}.{}".format(src, tgt, tgt))

            results = {}
            tasks_list = []

            cnt = 0
            with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
                for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                    src_sent = src_sent.strip()
                    tgt_sent = tgt_sent.strip()
                    tasks_list.append(process_sample(lang_pair, src_sent, tgt_sent, num_samples, temperature, top_p))

                    # Limit concurrency to 20 tasks at a time
                    if len(tasks_list) > 50:
                        completed, tasks = await asyncio.wait(tasks_list, return_when=asyncio.FIRST_COMPLETED)
                        for task in completed:
                            src_sent, tgt_sent, outputs = task.result()
                            results[src_sent] = [tgt_sent, outputs]
                        # Remove completed tasks from the list
                        tasks_list = list(tasks)

                    cnt += 1
                    if cnt % 100 == 0:
                        print(f"Process {cnt} samples ...")

            # Process remaining tasks after the loop
            remaining_results = await asyncio.gather(*tasks_list)
            for src_sent, tgt_sent, outputs in remaining_results:
                results[src_sent] = [tgt_sent, outputs]

            # Save results
            output_dir = os.path.join("dataset", "{}-{}-{}-{}-{}".format(data, model_name, str(num_samples), str(temperature), str(top_p)), lang_pair)
            os.makedirs(output_dir, exist_ok=True)
            cnt = 0
            with open(os.path.join(output_dir, "src"), 'w') as s_fout, \
                    open(os.path.join(output_dir, "tgt"), 'w') as t_fout, \
                    open(os.path.join(output_dir, "ref"), 'w') as r_fout:
                for sent in results:
                    src = sent
                    ref = results[sent][0]

                    if len(results[sent][1]) != num_samples:
                        print("WARN: Number Mismatch")
                        print(results[sent][1], flush=True)
                    else:
                        for tgt in results[sent][1]:
                            s_fout.write(src.strip() + "\n")
                            t_fout.write(tgt.strip() + "\n")
                            r_fout.write(ref.strip() + "\n")
                            cnt += 1
            print("Total sentence pairs: {}".format(cnt))


if __name__ == "__main__":
    asyncio.run(main())

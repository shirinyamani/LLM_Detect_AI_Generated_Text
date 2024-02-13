from tokenizers import (Tokenizer, decoders, models, normalizers,
                        pre_tokenizers, processors, trainers)
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, CodeGenTokenizer,
                          PreTrainedTokenizerFast)



llm_tokenizer = CodeGenTokenizer.from_pretrained("microsoft/phi-2", add_bos_token = True, trust_remote_code=True)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=dtype,
                                                    device_map=device, trust_remote_code=True)
max_length = 2048
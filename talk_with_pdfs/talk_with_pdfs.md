# Talk with your PDFs

Nowadays it is really simple to just upload your PDF to an online LLM and them about it? But how does it work?

Basically we can create one for ourselves with `llama-index`. 

The logic is the following:
* define your files
* vectorize them, alias turn them into small chunks, than generate an embedding for each chunk, either locally or using some online tool (but that would cost money)
* when you ask a question, the script retrives the most relevant docs's chunks and sends to Gemini LLM
* talk with your documents!

Note: I used gemini, because of the free PRO package available for students.

## Prerequisites.

Install the requirements from `requirements.txt`. (use venvs to not litter your global env)

```
pip install -r requirements.txt
```

Get your `GEMINI_API_KEY` from [Google's AIStudio](https://ai.google.dev/gemini-api/docs/api-key). Add to `secrets.env`.

Add your data to `/docs/`

Tests were ran using `Python 3.12.12` on MacOS.

### Hope this helps, have fun!
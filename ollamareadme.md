# bdlaws-3.6

A Bangladesh-law-only assistant for Bangla and English legal help.

## Run

```bash
ollama run shaqirii13/bdlaws-3.6
```

## Overview

`bdlaws-3.6` is an Ollama-ready model built specifically for Bangladesh law workflows. It is designed to:

- answer Bangladesh law questions in Bangla and English
- give short, law-focused introductions when greeted
- refuse off-topic requests such as math, weather, or general chat
- work best with retrieval-backed legal text pipelines

## What It Is For

This model is best suited for:

- Bangladesh law related questions
- Act, order, ordinance, and section lookup
- Plain-language explanation of Bangladeshi legal text
- Local legal assistants that combine retrieval with generation

## Behavior Policy

- Greeting-friendly: it can briefly introduce itself.
- Bangladesh-law only: it should not behave like a general assistant.
- Off-topic prompts: it should refuse and remind the user that it was built for Bangladesh law assistance.
- High-stakes answers: always verify with the original law text.

## Recommended Usage

For the best results, use this model with a retrieval pipeline over Bangladesh law texts so the answer stays grounded in the relevant act, section, or page.

Recommended workflow:

1. Retrieve the relevant law text or sections.
2. Pass the grounded context to the model.
3. Show citations in the final answer.
4. Verify important claims against the original legal source.

## Limitations

- Not a substitute for a lawyer or formal legal advice
- May produce incomplete or unverified answers without retrieval
- May reflect OCR or source-text quality issues from the legal corpus
- Not intended for jurisdictions outside Bangladesh law
- Broad prompts without act/section/year details may require clarification first

## Safety Note

This model should be treated as an assistive legal research tool, not as an authoritative legal decision-maker. For compliance, court, contract, rights, or liability-sensitive matters, always verify with the original law text or a qualified legal professional.

## Identity

This model is presented as `bdlaws-3.6` and identifies itself as a Bangladesh law assistant developed by Md Shakil Ahmed, a Bangladeshi AI researcher.

## Project

Open-source repository:

https://github.com/ahmedmshakil/bdlaws-3.5.git

## License

The repository code is released under the MIT License.  
Base model and upstream components remain subject to their original licenses and terms.

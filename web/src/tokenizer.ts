// GPT-2 byte-level BPE, the tokenizer the model was trained with (tiktoken's "gpt2" encoding).
// js-tiktoken/lite plus the gpt2 ranks is the whole vocabulary; no network request is needed.
import { Tiktoken } from 'js-tiktoken/lite';
import gpt2 from 'js-tiktoken/ranks/gpt2';

export const EOT = 50256; // <|endoftext|>, the token sample.py puts before every prompt

const encoder = new Tiktoken(gpt2);

/** Ordinary (non-special) BPE, as tiktoken's encode_ordinary. */
export function encode(text: string): number[] {
  return encoder.encode(text);
}

export function decode(tokens: number[]): string {
  return encoder.decode(tokens);
}

/**
 * Decode while tokens are still arriving. A token can hold part of a multi-byte
 * UTF-8 character, so the tail replacements from TextDecoder are held back until
 * the next token completes the character.
 */
export function decodePartial(all: number[], generated: number[]): string {
  const text = decode(all);
  let end = text.length;
  while (end > 0 && text.charCodeAt(end - 1) === 0xfffd) end--;
  return end === text.length ? text : decode(all.slice(0, generated.length - 1));
}

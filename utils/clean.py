import re


start_token = ''
end_token = ''
padding_token = ''


def _make_padding(seq_length):
    """
    Create a padding token of the given sequence length.
    """
    return "".join( [end_token] + [padding_token] * seq_length )

def cleanup_simple_wiki(text , seq_length):
    pad = _make_padding(seq_length)
    
    text = start_token+ re.sub(r'\n\n' , pad + start_token , text) + pad
    return text


def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text

def cleanup_bnc_spoken(text, seq_length):
    pad_seq = _make_padding(seq_length)
    text = cleanup_extra_spaces(text)
    text = start_token + re.sub(r'\n\n', pad_seq + start_token, text) + pad_seq
    return text


def cleanup_childes(text, seq_length):
    text = cleanup_extra_spaces(text)
    return start_token + text + _make_padding(seq_length)


def cleanup_gutenberg(text, seq_length):
    return text + ''.join(seq_length * [padding_token])

def cleanup_open_subtitles(text, seq_length):
    subtitle_credit_ex = re.compile(r'^.*subtitle.*$\n', re.MULTILINE | re.IGNORECASE)
    text = subtitle_credit_ex.sub('', text)
    return start_token + text + _make_padding(seq_length)


def cleanup_switchboard(text, seq_length):
    # No start or end tokens because the text seems to be cut.
    return text + ''.join(seq_length * [padding_token])


print(cleanup_bnc_spoken("This is a                  test.\n\nThis is a test2 oonc cricket", 5))
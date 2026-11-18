"""Brazilian Portuguese text normalizer: digits, money, dates, times, phone
numbers, ordinals, symbols and common abbreviations become words.

The same function runs on the training scripts (before the teacher TTS and the
phonemizer see them) and at inference, so the model never has to guess how a
digit is read. eSpeak NG reads some of these wrongly on its own: "R$ 19,90" as
"real dezenove vírgula noventa", "14h30" as "catorze agá trinta", dates as bare
numbers.

Pure Python, no dependencies.

Written for tiny-voca pt-BR (it has no upstream counterpart); Apache-2.0 like
the rest of this repository.
"""

import re
import unicodedata

UNITS = ("zero um dois três quatro cinco seis sete oito nove dez onze doze treze catorze quinze "
         "dezesseis dezessete dezoito dezenove").split()
TENS = ("_ _ vinte trinta quarenta cinquenta sessenta setenta oitenta noventa").split()
HUNDREDS = ("_ cento duzentos trezentos quatrocentos quinhentos seiscentos setecentos oitocentos novecentos").split()
MONTHS = ("janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro").split()
ORDINAL_UNITS = ("_ primeiro segundo terceiro quarto quinto sexto sétimo oitavo nono").split()
ORDINAL_TENS = ("_ décimo vigésimo trigésimo quadragésimo quinquagésimo sexagésimo septuagésimo octogésimo "
                "nonagésimo").split()
ORDINAL_HUNDREDS = ("_ centésimo ducentésimo trecentésimo quadringentésimo quingentésimo sexcentésimo "
                    "septingentésimo octingentésimo noningentésimo").split()

# Nouns that make a preceding number feminine ("duas pessoas", "duzentas casas").
FEMININE = frozenset("""
pessoa pessoas hora horas vez vezes semana semanas casa casas criança crianças mulher mulheres coisa coisas
página páginas palavra palavras cidade cidades empresa empresas opção opções pergunta perguntas resposta
respostas vaga vagas unidade unidades tonelada toneladas caixa caixas garrafa garrafas xícara xícaras colher
colheres fatia fatias maçã maçãs laranja laranjas banana bananas noite noites manhã manhãs tarde tardes aula
aulas nota notas estrela estrelas linha linhas letra letras música músicas foto fotos vida vidas parte partes
mesa mesas cadeira cadeiras porta portas janela janelas rua ruas escola escolas loja lojas moeda moedas conta
contas dica dicas ideia ideias história histórias frase frases etapa etapas família famílias filha filhas
amiga amigas menina meninas aluna alunas vitória vitórias derrota derrotas camisa camisas peça peças pizza
pizzas cerveja cervejas libra libras milha milhas volta voltas rodada rodadas fase fases temporada temporadas
edição edições questão questões regra regras lição lições hora-aula horas-aula receita receitas viagem viagens
mensagem mensagens chamada chamadas reunião reuniões tentativa tentativas mão mãos perna pernas folha folhas
cópia cópias sessão sessões consulta consultas parcela parcelas prestação prestações camiseta camisetas
caneta canetas empada empadas coxinha coxinhas vela velas bola bolas fruta frutas árvore árvores flor flores
praia praias bicicleta bicicletas moto motos semana-feira prova provas festa festas cama camas sala salas
""".split())

ABBREVIATIONS = {
    "sr.": "senhor", "sra.": "senhora", "srta.": "senhorita", "dr.": "doutor", "dra.": "doutora",
    "prof.": "professor", "profa.": "professora", "av.": "avenida", "etc.": "etcétera", "p.ex.": "por exemplo",
    "ltda.": "limitada", "pág.": "página", "págs.": "páginas", "tel.": "telefone", "cel.": "celular",
    "obs.": "observação", "aprox.": "aproximadamente", "vs.": "versus", "jr.": "júnior", "sto.": "santo",
    "sta.": "santa", "exmo.": "excelentíssimo", "exma.": "excelentíssima", "v.ex.ª": "vossa excelência",
}
UNIT_WORDS = {  # after a number only
    "km": ("quilômetro", "quilômetros"), "kg": ("quilo", "quilos"), "g": ("grama", "gramas"),
    "mg": ("miligrama", "miligramas"), "ml": ("mililitro", "mililitros"), "l": ("litro", "litros"),
    "m": ("metro", "metros"), "cm": ("centímetro", "centímetros"), "mm": ("milímetro", "milímetros"),
    "min": ("minuto", "minutos"), "s": ("segundo", "segundos"), "km/h": ("quilômetro por hora", "quilômetros por hora"),
    "gb": ("gigabyte", "gigabytes"), "mb": ("megabyte", "megabytes"), "°c": ("grau", "graus"), "°": ("grau", "graus"),
}
SYMBOLS = {"&": " e ", "+": " mais ", "=": " igual a ", "@": " arroba ", "§": " parágrafo ", "#": " hashtag "}


def _below_thousand(n, feminine=False):
    if n < 20:
        word = UNITS[n]
        return {"um": "uma", "dois": "duas"}.get(word, word) if feminine else word
    if n < 100:
        tens, unit = divmod(n, 10)
        return TENS[tens] + (" e " + _below_thousand(unit, feminine) if unit else "")
    if n == 100:
        return "cem"
    hundreds, rest = divmod(n, 100)
    word = HUNDREDS[hundreds]
    if feminine and word.endswith("entos"):
        word = word[:-5] + "entas"
    return word + (" e " + _below_thousand(rest, feminine) if rest else "")


def cardinal(n, feminine=False):
    """Integer to words. ``feminine`` changes um/dois and the hundreds."""
    if n < 0:
        return "menos " + cardinal(-n, feminine)
    if n < 1000:
        return _below_thousand(n, feminine)
    parts = []  # (three-digit coefficient, words)
    for size, one, many in ((10**12, "um trilhão", "trilhões"), (10**9, "um bilhão", "bilhões"),
                            (10**6, "um milhão", "milhões"), (1000, "mil", "mil")):
        count, n = divmod(n, size)
        if count:
            # milhão/bilhão are masculine nouns; only the thousands agree with the noun.
            parts.append((count, one if count == 1 else cardinal(count, feminine and size == 1000) + " " + many))
    if n:
        parts.append((n, _below_thousand(n, feminine)))
    if len(parts) == 1:
        return parts[0][1]
    # "e" before the last group when it is below 100 or a round hundred:
    # "mil e quinhentos", "dois milhões e quinhentos mil", but "mil duzentos e trinta".
    last = parts[-1][0]
    joiner = " e " if last < 100 or last % 100 == 0 else " "
    return " ".join(words for _, words in parts[:-1]) + joiner + parts[-1][1]


def ordinal(n, feminine=False):
    if n <= 0 or n >= 1000:
        return cardinal(n, feminine)
    parts = []
    hundreds, rest = divmod(n, 100)
    tens, unit = divmod(rest, 10)
    if hundreds:
        parts.append(ORDINAL_HUNDREDS[hundreds])
    if tens:
        parts.append(ORDINAL_TENS[tens])
    if unit:
        parts.append(ORDINAL_UNITS[unit])
    if feminine:
        parts = [p[:-1] + "a" for p in parts]
    return " ".join(parts)


def digits(text):
    return " ".join(UNITS[int(d)] for d in text if d.isdigit())


def _integer(text):
    return int(text.replace(".", ""))


def decimal(text, feminine=False):
    """"3,5" -> "três vírgula cinco"; "0,75" -> "zero vírgula setenta e cinco"."""
    whole, _, fraction = text.partition(",")
    words = cardinal(_integer(whole), feminine)
    if fraction:
        lead = len(fraction) - len(fraction.lstrip("0"))
        words += " vírgula " + " ".join(["zero"] * lead) + (" " if lead and fraction.strip("0") else "")
        words += cardinal(int(fraction)) if fraction.strip("0") else ("" if lead else "zero")
    return re.sub(r"\s+", " ", words).strip()


def money(whole, cents):
    reais = _integer(whole) if whole else 0
    centavos = int((cents + "0")[:2]) if cents else 0
    parts = []
    if reais:
        unit = "real" if reais == 1 else "reais"
        if reais >= 10**6 and reais % 10**6 == 0:
            unit = "de reais"
        parts.append(cardinal(reais) + " " + unit)
    if centavos:
        parts.append(cardinal(centavos) + (" centavo" if centavos == 1 else " centavos"))
    return " e ".join(parts) or "zero real"


def _money_match(match):
    whole, cents, scale = match.group(1), match.group(2), match.group(3)
    if not scale:
        return money(whole, cents)
    count = cardinal(_integer(whole))
    if cents == "5" or cents == "50":
        count += " " + scale + " e meio"
    elif cents:
        count = decimal(whole + "," + cents) + " " + scale
    else:
        count += " " + scale
    return count + (" reais" if scale == "mil" else " de reais")


def _fraction_match(match):
    gender = MEASURES.get(match.group(2).lower())
    if gender is None:
        return match.group(0)
    words = FRACTIONS[match.group(1)]
    return "meia" if words == "meio" and gender == "f" else words


def clock(hours, minutes):
    h, m = int(hours), int(minutes or 0)
    words = cardinal(h, feminine=True) if h else "zero"
    if not minutes or m == 0:
        return words + (" hora" if h == 1 or h == 0 else " horas")
    if m == 30:
        return words + " e meia"
    return words + " e " + cardinal(m)


def date(day, month, year=None):
    d, mo = int(day), int(month)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    words = ("primeiro" if d == 1 else cardinal(d)) + " de " + MONTHS[mo - 1]
    if year:
        y = int(year)
        if len(year) == 2:
            y += 2000 if y < 50 else 1900
        words += " de " + cardinal(y)
    return words


_MONEY = re.compile(r"R\$\s?(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{1,2}))?(?:\s?(milhões|milhão|bilhões|bilhão|mil)\b)?")
FRACTIONS = {"1/2": "meio", "1/3": "um terço", "2/3": "dois terços", "1/4": "um quarto", "3/4": "três quartos"}
# A fraction only before a measure word ("1/2 xícara"); otherwise "1/3" is a date.
MEASURES = {"xícara": "f", "xícaras": "f", "colher": "f", "colheres": "f", "hora": "f", "pizza": "f", "dúzia": "f",
            "porção": "f", "lata": "f", "copo": "m", "copos": "m", "kg": "m", "quilo": "m", "litro": "m", "pacote": "m",
            "tablete": "m", "cebola": "f", "maçã": "f", "laranja": "f", "limão": "m", "pão": "m", "metro": "m"}
_FRACTION = re.compile(r"(?<![\d/])(1/2|1/3|2/3|1/4|3/4)(?![\d/])(?=\s+(\w+))")
_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?\b")
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2})|h(\d{2})?(?:min)?)\b")
_PHONE = re.compile(r"(?:\(?\b(\d{2})\)?\s?)?\b(\d{4,5})-(\d{4})\b")
_CEP = re.compile(r"\b(\d{5})-(\d{3})\b")
_PERCENT = re.compile(r"\b(\d+(?:,\d+)?)\s?%")
_ORDINAL = re.compile(r"\b(\d+)\s?([ºª°])(?![CcFf])")
_NUMBER = re.compile(r"\b(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?\b")
_UNIT = re.compile(r"\b(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?\s?(km/h|°c|km|kg|mg|ml|cm|mm|min|gb|mb|°|g|l|m|s)(?![\wº])",
                   re.IGNORECASE)
_TIMES = re.compile(r"\b(\d+)\s?x\s?(\d+)\b")


def _next_word(text, end):
    match = re.match(r"\s*([A-Za-zÀ-ÿ-]+)", text[end:])
    return match.group(1).lower() if match else ""


def _numbers(text):
    def replace(match):
        whole, fraction = match.group(1), match.group(2)
        feminine = _next_word(text, match.end()) in FEMININE
        if fraction is not None:
            return decimal(whole + "," + fraction, feminine)
        if len(whole) > 1 and whole.startswith("0"):
            return digits(whole)
        return cardinal(_integer(whole), feminine)
    return _NUMBER.sub(replace, text)


def _abbreviations(text):
    def replace(match):
        word = match.group(0)
        expansion = ABBREVIATIONS.get(word.lower())
        if expansion is None:
            return word
        expansion = expansion.capitalize() if word[0].isupper() and match.start() == 0 else expansion
        return expansion + ("." if not text[match.end():].strip() else "")
    return re.sub(r"(?<![\w.])(?:[A-Za-zÀ-ÿ]+\.)+(?:ª)?", replace, text)


def normalize(text):
    """Return ``text`` with everything a reader would say spelled out in words."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("–", "-").replace("—", " — ")
    text = _abbreviations(text)
    text = re.sub(r"\bn[º°]\s?(?=\d)", "número ", text, flags=re.IGNORECASE)
    text = _MONEY.sub(_money_match, text)
    text = _CEP.sub(lambda m: digits(m.group(1)) + " " + digits(m.group(2)), text)
    text = _PHONE.sub(lambda m: ((cardinal(int(m.group(1))) + ", ") if m.group(1) else "")
                      + digits(m.group(2)) + ", " + digits(m.group(3)), text)
    text = _FRACTION.sub(_fraction_match, text)
    text = _DATE.sub(lambda m: date(m.group(1), m.group(2), m.group(3)) or m.group(0), text)
    text = _TIME.sub(lambda m: clock(m.group(1), m.group(2) or m.group(3)) if int(m.group(1)) < 24 else m.group(0), text)
    text = _PERCENT.sub(lambda m: decimal(m.group(1)) + " por cento", text)
    text = _ORDINAL.sub(lambda m: ordinal(int(m.group(1)), m.group(2) == "ª"), text)

    def unit(match):
        name, plural = UNIT_WORDS[match.group(3).lower()]
        whole, fraction = match.group(1), match.group(2)
        value = decimal(whole + "," + fraction) if fraction else cardinal(_integer(whole))
        one = fraction is None and _integer(whole) == 1
        return value + " " + (name if one else plural)
    text = _UNIT.sub(unit, text)
    text = _TIMES.sub(lambda m: cardinal(int(m.group(1))) + " por " + cardinal(int(m.group(2))), text)
    text = _numbers(text)
    for symbol, words in SYMBOLS.items():
        text = text.replace(symbol, words)
    return re.sub(r"\s+", " ", text).strip()


__all__ = ["normalize", "cardinal", "ordinal", "decimal", "money", "clock", "date", "digits"]

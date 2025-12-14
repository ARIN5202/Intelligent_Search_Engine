import dateutil.parser as dparser

parsed_date = dparser.parse("tommorrow", fuzzy=True)
formatted_date = parsed_date.strftime("%Y-%m-%d")
print(formatted_date)
from datetime import datetime, timedelta
from user_database import get_customers

def _run_rules_engine():
    customer_list = get_customers()
    current_date = datetime(year=2024, month=1, day=1)
    for kundenummer, _ in customer_list.items():
        if current_date - customer_list[kundenummer].fodselsdato > timedelta(weeks=52*55):
            print(f'Setting {kundenummer} status "kan_pensjoneres" True')
            customer_list[kundenummer].kan_pensjoneres = True

    for kundenummer, _ in customer_list.items():
        if customer_list[kundenummer].ansiennitet > timedelta(weeks=15*52):
            print(f'Setting vacation to 7 weeks for {kundenummer}')
            customer_list[kundenummer].maks_ferie = timedelta(weeks=7)
            
    return customer_list

def rules_engine(kundenummer: str):
    print("Running rules engine.")
    customer_list = _run_rules_engine()
    if kundenummer in customer_list.keys():
        return customer_list[kundenummer]
    else:
        return f"Cannot identify customer with kundenummer {kundenummer}."

from pydantic import BaseModel
from datetime import datetime

class Customer(BaseModel):
    """Information about a customer."""
    fornavn: str  
    etternavn: str
    personnummer: str
    kundenummer: str
    fodselsdato: datetime
    pensjonert: bool
    kan_pensjoneres: bool
    

def get_customers():
    customer = dict()
    customer['0202198000001'] = Customer(
        fornavn='Anna',
        etternavn='Anna',
        personnummer='00001',
        kundenummer='0202198000001',
        fodselsdato=datetime(year=1980, month=2, day=2),
        pensjonert=False,
        kan_pensjoneres=False
    )

    customer['0202196000002'] = Customer(
        fornavn='Bjorn',
        etternavn='Bjorn',
        personnummer='00002',
        kundenummer='0202196000002',
        fodselsdato=datetime(year=1960, month=2, day=2),
        pensjonert=True,
        kan_pensjoneres=True  # This field should change when rules engine runs.
    )

    customer['0202198000003'] = Customer(
        fornavn='Charley',
        etternavn='Charley',
        personnummer='00003',
        kundenummer='0202198000003',
        fodselsdato=datetime(year=1940, month=2, day=2),
        pensjonert=True,
        kan_pensjoneres=False
    )

    return customer

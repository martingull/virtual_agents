from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Customer(BaseModel):
    """Information about a customer."""
    fornavn: Optional[str] = Field(default=None, description="The first name of the person")
    etternavn: Optional[str] = Field(default=None, description="The last name of the person")
    personnummer: Optional[str] = Field(default=None, description="Person nummer")
    kundenummer: Optional[str] = Field(default=None, description="Kundenummer eller personnummer")
    fodselsdato: Optional[datetime] = Field(default=None, description="Date of birth")
    pensjonert: Optional[bool] = Field(default=None, description="This person is currently retired")
    kan_pensjoneres: Optional[bool] = Field(default=None, description="This person is eligable to retire")
    

def get_customers():
    customer_dict = dict()
    customer_dict['0202198000001'] = Customer(
        fornavn='Anna',
        etternavn='Anna',
        personnummer='00001',
        kundenummer='0202198000001',
        fodselsdato=datetime(year=1980, month=2, day=2),
        pensjonert=False,
        kan_pensjoneres=False
    )

    customer_dict['0202196000002'] = Customer(
        fornavn='Bjorn',
        etternavn='Bjorn',
        personnummer='00002',
        kundenummer='0202196000002',
        fodselsdato=datetime(year=1960, month=2, day=2),
        pensjonert=True,
        kan_pensjoneres=True  # This field should change when rules engine runs.
    )

    customer_dict['0202198000003'] = Customer(
        fornavn='Charley',
        etternavn='Charley',
        personnummer='00003',
        kundenummer='0202198000003',
        fodselsdato=datetime(year=1940, month=2, day=2),
        pensjonert=True,
        kan_pensjoneres=False
    )

    return customer_dict

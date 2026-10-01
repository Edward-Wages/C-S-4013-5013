import csv
import random
from collections.abc import Sequence
from pathlib import Path

from matplotlib import pyplot


class CreditCardApplication:
    """
    A row in "CreditCard.csv"
    """
    def __init__(
        self,
        id: str = "",
        credit_approve: bool = False,
        gender: int = 0,
        car_owner: int = 0,
        property_owner: int = 0,
        children: int = 0,
        work_phone: int = 0,
        email: int = 0,
    ) -> None:
        self.credit_approve = credit_approve
        self.gender = gender
        self.car_owner = car_owner
        self.property_owner = property_owner
        self.children = children
        self.work_phone = work_phone
        self.email = email

    @property
    def x(self):
        """
        Returns a tuple of the "meaningful" attributes which affect the error
        """
        return (self.gender, self.car_owner, self.property_owner, self.children, self.work_phone, self.email)

    def __repr__(self) -> str:
        return repr(self.__dict__)

class CreditCardApplications:
    """
    A representation of "CreditCard.csv"
    """
    def __init__(self, applications: dict[str, CreditCardApplication] | None = None) -> None:
        self.applications = {}
        if applications:
            self.applications = applications

    @staticmethod
    def from_csv(file_path="CreditCard.csv") -> "CreditCardApplications":
        applications = {}
        # Get the file path relative to the Python file instead of the current working directory
        path = Path(__file__).parent / file_path
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                applications[row["Ind_ID"]] = CreditCardApplication(
                    id=row["Ind_ID"],
                    credit_approve=bool(int(row["CreditApprove"])),
                    gender=int(row["Gender"] == "M"),
                    car_owner=int(row["CarOwner"] == "Y"),
                    property_owner=int(row["PropertyOwner"] == "Y"),
                    children=int(row["#Children"]),
                    work_phone=int(row["WorkPhone"]),
                    email=int(row["Email_ID"]),
                )

        return CreditCardApplications(applications)

    def __repr__(self) -> str:
        return repr(self.applications)

    def er(self, w: Sequence) -> float:
        """
        The error function to minimize
        """
        sum = 0
        for application in self.applications.values():
            f = 0
            for i in range(len(w)):
                f += w[i] * application.x[i]
            sum += (f - int(application.credit_approve))**2

        n = len(self.applications)
        return sum/n

def local_search():
    """
    Returns a generator of:
        (w, error)
    """
    applications = CreditCardApplications.from_csv()

    # Random 6-tuple of either -1 or 1
    w = [random.choice([-1, 1]) for _ in range(6)]
    err = applications.er(w)
    yield (w, err)
    while True:
        local_w = w
        local_err = err
        for i in range(len(w)):
            neighbor = w.copy()
            neighbor[i] *= -1
            neighbor_err = applications.er(neighbor)
            if neighbor_err < local_err:
                local_err = neighbor_err
                local_w = neighbor
        if local_err < err:
            w = local_w
            err = local_err
            yield (w, err)
        else:
            break

# Run the algorithm a hundred times for a long sequence, but an ideal error
best_result = [(None, 10)]
for i in range(100):
    result = list(local_search())
    if result[-1][1] <= best_result[-1][1] and len(result) >= len(best_result):
        best_result = result

fig, ax = pyplot.subplots()
ax.plot([x[1] for x in best_result])
fig.savefig(Path(__file__).parent / "local_search.png")
pyplot.show()

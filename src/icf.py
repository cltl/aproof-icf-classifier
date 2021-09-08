import enum

output_columns = ["D450 (FAC)", "B440 (ADM)", "B140 (ATT)", "D840-D859 (BER)", "B1300 (ENR)", "D550 (ETN)",
                  "B455 (INS)", "B530 (MBW)", "disregard", "target"]

output_values = ["0-5", "0-4", "0-4", "0-4", "0-4", "0-4", "0-5", "0-4", False, ""]

class ICF(enum.Enum):
    D450 = 0
    B440 = 1
    B140 = 2
    D840D859 = 3
    B1300 = 4
    D550 = 5
    B455 = 6
    B530 = 7
    DISREGARD = 8
    TARGET = 9


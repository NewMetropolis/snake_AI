# # MN Wygląda jak zakończenie komentarza "z góry"(?).
# # W każdym razie nie powinno tu tego być.
# # """
# # MN Definicja zmiennej przed importem modułów. Samo nazewnictwo także niepoprawne.
# # Nazwa zmiennej powinna odnosić się do jej charakteru/roli.
# foo = []
# # MN Zbędny Import modułu, który nie jest wykorzystywany.
# # from .models import Expense
# # MN Literówka. Nazwa biblioteki to 'collections'.
# # from collection import namedtuple
# from collections import namedtuple
#
# # MN Mało czytelne nazewnictwo: typename = 'F'.
# # Odbiega od zalecanej przez autorów konwencji. W tym przypadku, typename = 'MyExpense'.
# MyExpense = namedtuple('F', ['type_', 'amount'])
#
# history = [['food', 4], ['food', 3], ['car', 3], ['dog', 1]]
# # foo = list(map(MyExpense._make, history))
# # 'Kod spagetti'. Można by np. użyć '_make' + map().
# foo.append(MyExpense('food', 4))
# foo.append(MyExpense('food', 3))
# foo.append(MyExpense('car', 3))
# foo.append(MyExpense('dog', 1))

#
# # MN Kolejna niezgodność z PEP. W nazwach funkcji nie używa się camelCase.
# # MN Nazwa argumentu 'input' pokrywa się z wbudowaną funkcją 'input'.
# def summarize_expenses(min_amount, input):
#     expenses = {}
#     for expense in input:
#         if expense.amount >= min_amount:
#             # MN PEP zaleca składnie 'not in'.
#             if expense.type_ not in expenses:
#                 expenses[expense.type_] = 0
#             expenses[expense.type_] = expenses[expense.type_] + expense.amount
#     # MN Ponownie zamieszanie w nazewnictwie, for(type_, amount) powinno sprawdzić się lepiej.
#     # Patrz także komentarz przy 'print'.
#     for (type_, amount) in sorted(expenses.items(), key=lambda e: e[1], reverse=False):
#         # Brakuje nawiasów, print(expense.type_, amount). Ewentualnie z wykorzystniem .format().
#         # Po posortwoaniu elementów słownika otrzymujemy liste tupli, które następnie odpakowujemy.
#         # Po tych operacjach 'expense' to po prostu string. Użycie 'type_' w miejsce 'expense' powinno rozwiązać
#         # dwuznaczności.
#         print(type_, amount)
#     # Ja zawsze używam return
#
#
# summarize_expenses(2, foo)


from collections import namedtuple


def summarize_expenses(min_amount, expenses):
    expenses_by_cat = {}
    for expense in expenses:
        if expense.amount >= min_amount:
            if expense.type_ not in expenses_by_cat:
                expenses_by_cat[expense.type_] = 0
            expenses_by_cat[expense.type_] += expense.amount

    for (type_, amount) in sorted(expenses_by_cat.items(), key=lambda e: e[1], reverse=False):
        print('{}: {}'.format(type_, amount))

    return


MyExpense = namedtuple('MyExpense', ['type_', 'amount'])
history = [['food', 4], ['food', 3], ['car', 3], ['dog', 1]]
expenses_list = list(map(MyExpense._make, history))


summarize_expenses(2, expenses_list)

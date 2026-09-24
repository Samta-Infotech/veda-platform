# Questions that answer on the local main pipeline

Measured 60 of test.py's questions, one fresh chat each. 27 answered with SQL; 33 refused or asked for clarification.

These are the ones the drill-down and memory passes were run on.

| # | category | question | rows |
|---:|---|---|---:|
| 1 | ledger | Show the latest 10 ledger entries. | 10 |
| 2 | property_db | How many properties allow all day access? | 100 |
| 3 | property_db | Count transactions where exchange rate is greater than 74. | 1 |
| 4 | property_db | How many payment transaction history records exist? | 100 |
| 5 | shapes | What is the distribution of properties by furnishing? | 12 |
| 6 | shapes | What is the distribution of properties by facing? | 18 |
| 7 | shapes | What is the distribution of lease listings by status? | 3 |
| 8 | shapes | What is the distribution of payment transactions by transaction type? | 6 |
| 9 | shapes | Which lease listings have the highest expected monthly rent? | 100 |
| 10 | shapes | Which payment transactions have the highest paid amount? | 100 |
| 11 | shapes | What is the total expected monthly rent across all lease listings? | 1 |
| 12 | shapes | What is the average expected price of sale listings? | 1 |
| 13 | shapes | Show properties where is gated is true. | 1000 |
| 14 | shapes | List properties with furnishing 'FULL'. | 1000 |
| 15 | shapes | List sale listings with status 'APPROVED'. | 195 |
| 16 | shapes | How many sale listings are there? | 1 |
| 17 | shapes | List all amenities. | 32 |
| 18 | shapes | What are the names of all projects? | 1000 |
| 19 | shapes | show lease listings and their property | 1000 |
| 20 | shapes | show properties with their amenities | 0 |
| 21 | shapes | top 5 properties by number of payment transactions | 5 |
| 22 | shapes | list all tenants | 32 |
| 23 | shapes | List all amenity categories. | 3 |
| 24 | shapes | List all payment types. | 6 |
| 25 | business | What is the total outstanding amount for all maintenance items that are not yet completed or settled? | 1 |
| 26 | business | Which category contributes the highest value among all completed payments? | 0 |
| 27 | business | How much money has already been collected through completed payments? | 100 |

## Refused, grouped by the reason the engine gave

### could not map an ordinary English word to a column (12)

- **Show the 10 transactions made by the payer "Tenant" with transaction details such as receiver name, entry type, date, amount, and currency.**
  - I couldn't map 'currency' to any column or value in the data.
   → Tell me what 'currency' refers to — a column, or a va
- **Display all transactions from the ledger where the transaction amount exceeds 10,000.**
  - I couldn't map 'exceed' to any column or value in the data.
   → Tell me what 'exceed' refers to — a column, or a value 
- **How many tickets are there in each status category?**
  - I couldn't map 'statu' to any column or value in the data.
   → Tell me what 'statu' refers to — a column, or a value to
- **What is the average time taken to resolve closed tickets?**
  - I couldn't map 'average' to any column or value in the data.
   → Tell me what 'average' refers to — a column, or a valu
- **How many properties are listed in the database?**
  - I couldn't map 'database' to any column or value in the data.
   → Tell me what 'database' refers to — a column, or a va
- **Which floor has the maximum number of properties?**
  - I couldn't map 'property' to any column or value in the data.
   → Tell me what 'property' refers to — a column, or a va
- **How many properties are in a gated community vs non-gated community?**
  - I couldn't map 'non' to any column or value in the data.
   → Tell me what 'non' refers to — a column, or a value to fil
- **How many ledger transactions exist?**
  - I couldn't map 'exist' to any column or value in the data.
   → Tell me what 'exist' refers to — a column, or a value to
- **How many settlement history records exist?**
  - I couldn't map 'exist' to any column or value in the data.
   → Tell me what 'exist' refers to — a column, or a value to
- **What is the monthly trend of properties based on created at?**
  - I couldn't map 'property' to any column or value in the data.
   → Tell me what 'property' refers to — a column, or a va
- **What is the average carpet area sqft of properties?**
  - I couldn't map 'average' to any column or value in the data.
   → Tell me what 'average' refers to — a column, or a valu
- **show properties with their owners**
  - I couldn't map 'property' to any column or value in the data.
   → Tell me what 'property' refers to — a column, or a va

### could not work out a reliable total (6)

- **How many properties have power backup vs no power backup?**
  - I couldn't work out a reliable total for this, so I'd rather not show a figure that might be wrong
   → Try asking for o
- **How many properties are corner properties?**
  - I couldn't work out a reliable total for this, so I'd rather not show a figure that might be wrong
   → Try asking for o
- **What is the total settlement expected, paid, and balance amount?**
  - I couldn't work out a reliable total for this, so I'd rather not show a figure that might be wrong
   → Try asking for o
- **How many properties were there in the last 12 months by created at?**
  - I couldn't work out a reliable total for this, so I'd rather not show a figure that might be wrong
   → Try asking for o
- **How many properties are there?**
  - I couldn't work out a reliable total for this, so I'd rather not show a figure that might be wrong
   → Try asking for o
- **How many projects do we have?**
  - I couldn't work out a reliable total for this, so I'd rather not show a figure that might be wrong
   → Try asking for o

### ranking guard (added today - was previously a wrong answer) (4)

- **Show top 5 highest CREDIT transactions**
  - you asked for the top 5 by value, but the query I built returns the rows in no particular order — rank by expected_amoun
- **Show top 5 highest DEBIT transactions**
  - you asked for the top 5 by value, but the query I built returns the rows in no particular order — rank by expected_amoun
- **Can you show me the 10 most recently created tickets along with their details?**
  - you asked for the top 10 by value, but the query I built returns the rows in no particular order — please name the colum
- **What percentage of vendors have ratings of at least 4.0?**
  - which amount should be ranked — max_rating, rating?
   → Could you tell me which one you mean?

### could not apply the condition (4)

- **How many lease transactions are recorded with currency ID 30?**
  - I couldn't apply the condition you asked for, so I'd rather not show numbers that ignore it
   → Try naming the field an
- **Show all asset types.**
  - I couldn't apply the condition you asked for, so I'd rather not show numbers that ignore it
   → Try naming the field an
- **top 5 payment transactions by paid amount**
  - I couldn't apply the condition you asked for, so I'd rather not show numbers that ignore it
   → Try naming the field an
- **top 10 users by total paid amount**
  - I couldn't apply the condition you asked for, so I'd rather not show numbers that ignore it
   → Try naming the field an

### other (3)

- **How many new properties have been added after the year 2012?**
  - The provided documents do not contain any information about properties or additions made after the year 2012. The contex
- **What percentage of all maintenance records are related to repair work?**
  - The provided context does not contain information about the percentage of maintenance records related to repair work. So
- **Which assets are currently covered under the maintenance arrangement?**
  - This policy governs asset 21 and asset 22 maintenance, including Football court and Basket Ball court upkeep in Muddanah

### generic clarify (federated refusal wording) (2)

- **Show top five general ledger entries**
  - Could you clarify what you're asking about?
- **Which properties have the highest carpet area?**
  - Could you clarify what you're asking about?

### ambiguous - asked instead of guessing (2)

- **Which city has the highest-rated vendor, and what is the vendor's rating?**
  - This question could apply to more than one thing in the data, so I'd rather ask than guess.
   → Could you tell me which
- **What is the average vendor rating across all service locations?**
  - This question could apply to more than one thing in the data, so I'd rather ask than guess.
   → Could you tell me which


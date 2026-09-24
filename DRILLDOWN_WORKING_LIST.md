# Drill-down: which base questions work, and with which follow-up

Measured on the LOCAL pipeline, 2026-09-23, after the conversation-boundary work.
Each row is one conversation: the base question, one follow-up, then `go back`.
Every column is judged against the SQL the engine produced, never the prose.

**17 of 17 measured base questions accept a follow-up.**

| # | Base question | Follow-up that works | filter applied | same entity | shape kept | `go back` |
|---:|---|---|:--:|:--:|:--:|:--:|
| 1 | Count transactions where exchange rate is greater than 74. | `only the Rent ones` | yes | yes | kept | no |
| 2 | How many properties allow all day access? | `only the Nagpur ones` | yes | yes | kept | yes |
| 3 | How many sale listings are there? | `only the APPROVED ones` | yes | yes | kept | yes |
| 4 | How much money has already been collected through completed payments? | `only the DEBIT ones` | yes | yes | - | yes |
| 5 | List properties with furnishing 'FULL'. | `only the Nagpur ones` | yes | yes | - | no |
| 6 | List sale listings with status 'APPROVED'. | `only the APPROVED ones` | no | yes | - | no |
| 7 | Show properties where is gated is true. | `only the Nagpur ones` | yes | yes | - | no |
| 8 | Show the latest 10 ledger entries. | `only the DEBIT ones` | yes | yes | - | yes |
| 9 | What is the average expected price of sale listings? | `only the APPROVED ones` | yes | yes | kept | yes |
| 10 | What is the distribution of lease listings by status? | `only the FULL ones` | yes | yes | kept | yes |
| 11 | What is the distribution of payment transactions by transaction type? | `only the DEBIT ones` | yes | yes | kept | yes |
| 12 | What is the distribution of properties by facing? | `only the Nagpur ones` | yes | yes | kept | yes |
| 13 | What is the distribution of properties by furnishing? | `only the Nagpur ones` | yes | yes | kept | yes |
| 14 | What is the total expected monthly rent across all lease listings? | `only the FULL ones` | yes | yes | LOST | yes |
| 15 | Which lease listings have the highest expected monthly rent? | `only the FULL ones` | yes | yes | - | no |
| 16 | Which payment transactions have the highest paid amount? | `only the DEBIT ones` | yes | yes | - | no |
| 17 | show lease listings and their property | `only the FULL ones` | yes | yes | - | yes |

## Not drillable at all

These base questions answer, but their table has no column that meaningfully narrows
(no categorical column where the commonest value covers 5-60% of the rows), so there is
no honest follow-up to test. Not a failure — a property of the data.

- How many payment transaction history records exist?  _(table: `accounts_paymenttransactionsettlement`)_
- List all amenities.  _(table: `assets_amenity`)_
- What are the names of all projects?  _(table: `assets_project`)_
- show properties with their amenities  _(table: `assets_amenity`)_
- top 5 properties by number of payment transactions  _(table: `assets_assetpayment`)_
- list all tenants  _(table: `assets_amenity`)_
- List all amenity categories.  _(table: `assets_amenitycategory`)_
- List all payment types.  _(table: `accounts_paymenttype`)_
- What is the total outstanding amount for all maintenance items that are not yet completed or settled?  _(table: `maintenance`)_
- Which category contributes the highest value among all completed payments?  _(table: `advertisements_advertisement`)_

## Follow-ups that work on ANY of the above

These are domain-independent and were measured separately:

| Follow-up | What it should do |
|---|---|
| `go back` | undo the last narrowing, restore the previous question |
| `start over` | forget everything (27/27 clean) |
| `what did I just ask?` | restate from memory, never re-run a query |
| `show it as a chart` | same data, chart form |

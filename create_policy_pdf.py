import fitz  # PyMuPDF
import os

def create_enterprise_policy_pdf(output_path="corporate_travel_and_expense_policy.pdf"):
    """
    Generates a realistic, structured, 4-page enterprise policy PDF document.
    Includes general rules, specific exception clauses, negation constraints,
    lexical distractor traps, and buried penalty provisions.
    """
    doc = fitz.open()

    # Document sections
    pages_content = [
        # PAGE 1
        {
            "title": "GLOBAL TRAVEL, EXPENSE AND PROCUREMENT POLICY (2026)",
            "subtitle": "Acme Global Enterprise Operations & Compliance Manual — Document ID: POL-FIN-2026-v4.2",
            "sections": [
                (
                    "SECTION 1: PURPOSE, GOVERNANCE AND PRE-TRIP AUTHORIZATIONS",
                    [
                        ("1.1 Purpose and Universal Scope",
                         "This policy governs all business travel, entertainment, and procurement expenses incurred on behalf of Acme Global Corporation. Adherence is mandatory for all regular full-time employees, contractors, temporary workers, and executive officers. Any deviation requires written pre-approval as defined herein."),
                        ("1.2 Mandatory Pre-Trip Travel Authorization",
                         "Employees must secure written pre-authorization via the Travel Authorization Portal (TAP) at least 14 calendar days prior to domestic travel and 21 calendar days prior to international departures. Expenses incurred without prior electronic manager sign-off are non-reimbursable."),
                        ("1.3 Designated Corporate Travel Booking Engine",
                         "All commercial airfare, train tickets, rental cars, and hotel reservations must be booked exclusively through the enterprise corporate booking system (Concur Travel). Self-booking through external commercial websites (e.g., Expedia, Booking.com, direct airline portals) is prohibited and non-reimbursable except under declared catastrophic emergency disruptions approved by the Travel Operations team.")
                    ]
                ),
                (
                    "SECTION 2: AIR TRAVEL STANDARDS, UPGRADES AND CABIN CLASSES",
                    [
                        ("2.1 Standard Cabin Class Mandate",
                         "All domestic and international air travel must be booked exclusively in Standard Economy (Coach) class, regardless of employee seniority, managerial title, or total travel duration. Preferred economy seating (e.g., extra legroom, exit rows) is permitted only if at zero incremental charge to the corporate account or as an officially documented medical accommodation."),
                        ("2.2 Frequent Flyer Upgrades and Miles Redemption",
                         "Employees may apply personal frequent flyer status, miles, or personal funds to upgrade flight segments. However, the corporate reimbursement is strictly limited to the lowest non-refundable Standard Economy fare available at the time of reservation. The company does not reimburse frequent flyer membership fees or elite status subscription programs."),
                        ("2.3 Checked Luggage and Baggage Allowance",
                         "The company will reimburse a maximum of one (1) standard checked bag per traveler for domestic and continental flights lasting up to 5 days. For international travel exceeding 7 days, up to two (2) checked bags are reimbursable. Overweight bag fees and sporting equipment surcharges are the sole personal responsibility of the traveler.")
                    ]
                )
            ]
        },
        # PAGE 2
        {
            "title": "AIR TRAVEL EXCEPTIONS & LODGING REGULATIONS",
            "subtitle": "Document ID: POL-FIN-2026-v4.2 — Page 2 of 4",
            "sections": [
                (
                    "SECTION 2 (CONTINUED): LONG-HAUL EXCEPTIONS AND CABIN UPGRADES",
                    [
                        ("2.4 Long-Haul International Travel Exception Clause",
                         "Notwithstanding the strict Economy Class mandate in Section 2.1, employees traveling on continuous non-stop international flights with scheduled flight duration exceeding 8 hours (or exceeding 6 hours if the traveler is scheduled to deliver a client presentation or attend executive contract negotiations within 12 hours of flight arrival) are eligible to book Business Class. All such exceptions require prior written concurrence from both the Division Vice President and the Chief Financial Officer at least 10 business days prior to ticketing."),
                        ("2.5 Flight Cancellations, Disruptions and Refundable Fares",
                         "Fully refundable airfare is strictly disallowed unless travel dates are subject to court subpoenas or client regulatory mandates. In the event of airline schedule cancellations, travelers must accept airline rebooking or travel credits, which must be registered with Corporate Travel within 48 hours for future business redeployment.")
                    ]
                ),
                (
                    "SECTION 3: HOTEL LODGING, INCIDENTAL EXPENSES AND ROOM RESTRICTIONS",
                    [
                        ("3.1 Lodging Rate Ceilings and Preferred Hotel Chains",
                         "Hotel stays must be booked with designated preferred corporate hotel partners. Standard single occupancy room rates must not exceed the following nightly ceilings: Tier 1 Metros (New York, San Francisco, London, Tokyo, Zurich): $280/night; Tier 2 Metros (Chicago, Austin, Berlin, Singapore): $200/night; All other global locations: $150/night. Taxes and mandatory urban facility fees are excluded from these caps."),
                        ("3.2 Prohibited Hotel Incidentals and Personal Amenities",
                         "The corporation does not reimburse incidental expenses incurred during lodging. Non-reimbursable items include, but are not limited to: in-room video on-demand/movie rentals, mini-bar consumption, laundry service for trips shorter than 5 days, spa services, sauna access, dry cleaning fees, and hotel gym or fitness center pass fees."),
                        ("3.3 Lodging Cancellation Deadlines and No-Show Liability",
                         "Travelers are strictly responsible for cancelling hotel reservations within the hotel property's penalty-free cancellation window (typically 48 hours prior to check-in). Any no-show charges or penalty fees arising from employee negligence or failure to cancel reservations will be deducted from employee expense reports or charged directly to the traveler.")
                    ]
                )
            ]
        },
        # PAGE 3
        {
            "title": "GROUND TRANSPORTATION, MEALS & ENTERTAINMENT RESTRICTIONS",
            "subtitle": "Document ID: POL-FIN-2026-v4.2 — Page 3 of 4",
            "sections": [
                (
                    "SECTION 4: GROUND TRANSPORTATION, TAXIS AND RIDESHARING",
                    [
                        ("4.1 Commercial Ridesharing and Taxi Transfers",
                         "Employees are expected to use commercial ridesharing services (Uber, Lyft) or airport taxis for ground transit between airport terminals, hotels, and business meeting sites. Standard UberX or Lyft Standard rides are reimbursable up to $45 per one-way airport transfer without itemized route auditing. Public transit (metro, commuter train, airport express) is strongly encouraged where feasible."),
                        ("4.2 Rental Vehicles and Insurance Requirements",
                         "Rental cars should only be procured when public transportation or ridesharing is unavailable or demonstrably cost-ineffective. Approved vehicle category is limited to Compact or Intermediate sedans. Corporate travelers must decline supplementary collision damage waiver (CDW) as corporate fleet coverage applies globally."),
                        ("4.3 Personal Automobile Mileage Reimbursement",
                         "Business travel using personal vehicles is reimbursed at the prevailing standard rate of $0.67 per mile. Commuting mileage between the employee's residence and official office location is strictly non-reimbursable."),
                        ("4.4 Luxury Chauffeured Vehicles and Limousine Prohibition",
                         "Private chauffeured cars, black sedan executive services, town cars, and stretch limousines are strictly prohibited and non-reimbursable for airport transfers, local client visits, or routine business travel, even if booked via standard mobile applications (e.g., Uber Black, Lyft Black, executive sedan dispatch). An exception is permitted exclusively for transporting visiting head-of-state dignitaries or prospective Board of Directors nominees, subject to explicit advance approval by the Chief Executive Officer.")
                    ]
                ),
                (
                    "SECTION 5: BUSINESS ENTERTAINMENT, CLIENT DINNERS AND ALCOHOL",
                    [
                        ("5.1 Client Entertainment and Prospective Partner Hosting",
                         "Employees hosting prospective clients, strategic partners, or external corporate vendors may expense reasonable business meals. The primary purpose of the gathering must be active commercial dialogue, and all attendee names, company affiliations, and project codes must be explicitly detailed on the expense submission."),
                        ("5.2 Meal Per-Diem Allowances and Gratuities",
                         "Standard daily per-diem rates for employee meals are capped at $75 per day for domestic travel and $100 per day for international destinations. Tips and gratuities up to 20% of the food and beverage subtotal are reimbursable. Gratuities exceeding 20% are the personal responsibility of the employee."),
                        ("5.3 Prohibited Entertainment Items and Alcohol Consumption Limits",
                         "The following expenses are strictly non-reimbursable and explicitly prohibited: (a) Alcoholic beverages exceeding one (1) standard single serving of wine or beer per person during authorized client dinners; alcoholic drinks during solo employee dining are 100% non-reimbursable under all circumstances. (b) Adult entertainment venues, nightclub admission or table bottle service fees, social club charges, and gentlemen's club invoices. (c) Tobacco, vape products, cannabis, and casino gambling losses. (d) Meals or travel for spouses, partners, dependents, or personal companions accompanying the traveler.")
                    ]
                )
            ]
        },
        # PAGE 4
        {
            "title": "EXPENSE REPORTING, AUDITS & PENALTIES FOR NON-COMPLIANCE",
            "subtitle": "Document ID: POL-FIN-2026-v4.2 — Page 4 of 4",
            "sections": [
                (
                    "SECTION 6: IT HARDWARE AND PROCUREMENT CONTROLS",
                    [
                        ("6.1 Hardware and Peripheral Acquisition",
                         "All computing equipment, laptops, smartphones, monitors, and peripherals must be procured through the Global IT Hardware Catalog. Personal purchases of electronics on personal credit cards will not be reimbursed."),
                        ("6.2 Software Subscriptions and Cloud Services",
                         "Employees are strictly prohibited from expensing recurring software licenses, generative AI platform subscriptions, or cloud compute resources on corporate expense cards without pre-approval from Enterprise SecOps.")
                    ]
                ),
                (
                    "SECTION 7: PROFESSIONAL CONFERENCES AND SEMINARS",
                    [
                        ("7.1 Conference Attendance and Continuing Education",
                         "Conference registration fees require approval from the Senior Director of the employee's business unit. Annual professional development expenditure is capped at $1,500 per individual employee per fiscal year.")
                    ]
                ),
                (
                    "SECTION 8: EXPENSE REPORTING TIMELINES, AUDITS AND PENALTIES",
                    [
                        ("8.1 Standard Expense Filing Timelines",
                         "Employees are required to compile and submit complete expense reports with legible itemized receipts within 14 calendar days of concluding any business travel or incurring an out-of-pocket business expense."),
                        ("8.2 Mandatory Receipt Itemization Thresholds",
                         "Itemized receipts displaying transaction date, merchant identification, items purchased, and payment method are mandatory for all expenditures exceeding $25. Credit card summary slips that do not break down purchased items are not acceptable documentation."),
                        ("8.3 Corporate Credit Card Compliance and Commingling",
                         "Corporate credit cards must be utilized strictly for legitimate business expenditures. Commingling personal expenditures on corporate charge cards is an explicit compliance infraction. Inadvertent personal charges must be identified and settled within 7 days."),
                        ("8.4 Post-Payment Audit Protocols",
                         "The Corporate Internal Audit team executes randomized automated audits across 15% of all filed expense reports monthly. Non-compliant claims identified during audits will be deducted from employee payroll."),
                        ("8.5 Late Submission Forfeiture Clause and Card Revocation Penalties",
                         "Expense reports submitted more than 30 calendar days after the conclusion of travel are subject to mandatory automatic rejection and permanent forfeiture. Under no circumstances will untimely expenses beyond 30 days be reimbursed by corporate funds. Furthermore, any employee who incurs two (2) or more 30-day late submissions within a single rolling fiscal year will face immediate and permanent revocation of their corporate credit card privileges, mandatory referral to People Operations for formal disciplinary action, and suspension of all future travel privileges.")
                    ]
                )
            ]
        }
    ]

    for p_idx, page_data in enumerate(pages_content):
        page = doc.new_page(width=595, height=842)  # A4 size
        
        # Header banner
        page.draw_rect(fitz.Rect(36, 30, 559, 85), color=(0.1, 0.2, 0.45), fill=(0.93, 0.95, 0.98), width=1)
        page.insert_text(fitz.Point(46, 52), page_data["title"], fontsize=13, fontname="helv", color=(0.1, 0.2, 0.5))
        page.insert_text(fitz.Point(46, 72), page_data["subtitle"], fontsize=8, fontname="helv", color=(0.3, 0.35, 0.4))
        
        y_cursor = 105
        
        for sec_title, subsections in page_data["sections"]:
            # Section header
            page.draw_rect(fitz.Rect(36, y_cursor, 559, y_cursor + 20), color=(0.8, 0.85, 0.9), fill=(0.88, 0.92, 0.97))
            page.insert_text(fitz.Point(44, y_cursor + 14), sec_title, fontsize=10, fontname="helv", color=(0.05, 0.15, 0.35))
            y_cursor += 28
            
            for sub_title, body in subsections:
                # Subsection title
                page.insert_text(fitz.Point(44, y_cursor), sub_title, fontsize=9, fontname="helv", color=(0.15, 0.2, 0.3))
                y_cursor += 12
                
                # Subsection body text (wrapped in rect)
                text_rect = fitz.Rect(44, y_cursor, 550, y_cursor + 70)
                rc = page.insert_textbox(text_rect, body, fontsize=8, fontname="helv", color=(0.2, 0.2, 0.2), lineheight=1.25)
                # Estimate space used
                # insert_textbox returns remaining height or negative if failed, let's advance cursor based on text length
                approx_lines = (len(body) // 95) + 1
                y_cursor += max(approx_lines * 11 + 10, 28)
                
            y_cursor += 10
            
        # Footer
        page.draw_line(fitz.Point(36, 810), fitz.Point(559, 810), color=(0.7, 0.7, 0.7), width=0.5)
        page.insert_text(fitz.Point(44, 825), "CONFIDENTIAL — ACME GLOBAL COMPLIANCE INTERNAL DOCUMENT", fontsize=7, fontname="helv", color=(0.5, 0.5, 0.5))
        page.insert_text(fitz.Point(490, 825), f"Page {p_idx + 1} of {len(pages_content)}", fontsize=7, fontname="helv", color=(0.5, 0.5, 0.5))

    doc.save(output_path)
    doc.close()
    print(f"Successfully generated complex enterprise policy PDF: {output_path} ({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    create_enterprise_policy_pdf()

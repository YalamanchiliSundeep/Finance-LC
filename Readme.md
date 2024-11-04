## Financial Document Management Application
This application is designed to streamline document management, search, and analysis for finance teams by automating metadata extraction and retrieval processes. With AI-powered tools, it reduces manual data entry, ensures compliance, and makes document workflows more efficient.

## Key Features
1. Streamlined Document Upload and Extraction
Supported File Types: Upload PDFs (text and scanned), Word documents, and images (JPEG, PNG).
Automatic Text Extraction: Extracts text from documents, even scanned images, using Optical Character Recognition (OCR). (Note: OCR for scanned images is planned as a Phase II feature).
2. Automated Metadata Extraction
# GPT-4 Metadata Extraction: Automatically identifies key financial details, such as:
Issuer: Tracks the issuing financial institution.
Beneficiary: Identifies the receiving party.
Amount: Captures payment amounts for quick reference.
Expiration Date: Monitors document deadlines.
Additional Details: Contract Number, Project Name, Purpose, Cancellation, and Renewal Terms.
Formatted Output: Extracted data is structured for quick review, minimizing manual data entry.
3. Efficient Document Search and Retrieval
FAISS-powered Search: Find relevant documents by specific financial terms, queries, or document contents.
Natural Language Querying: Supports queries like “Show documents for Project X” or “What is the renewal policy on Letter of Credit Y?”
Concise Answers with GPT-4: Retrieves documents and provides answers based on content.
4. Metadata and Document Management
Metadata Table: Stores metadata in a structured table, making it easy to track documents by project, expiration date, issuer, and more.
CSV Export: Saves metadata to a CSV for future reference or audits.
Manual Metadata Entry and Editing: Allows for easy additions or modifications, and deletion of outdated records.
5. Risk Management and Compliance
Expiration Date Tracking: Automates tracking of expiration dates, renewal policies, and cancellation terms to avoid missed deadlines.
Centralized Repository: Maintains contract and policy details in a searchable, accessible repository to support compliance and audits.
6. User-Friendly Streamlit Interface
Intuitive Dashboard: Simple, user-friendly dashboard for uploading, viewing, searching, and managing financial documents.
Streamlined Workflow: All interactions happen in one place, minimizing the learning curve and maximizing team productivity.
## How It Fits into Finance Operations
Document Management: Organize and manage letters of credit, agreements, contracts, and policies in one repository.
Quick Retrieval: AI-powered search allows fast access to specific financial details, saving time.
Audit Preparation: Extracted metadata facilitates quick reviews during audits, making it easy to confirm terms and validate contents.
Proactive Deadline Monitoring: Tracks expiration dates and terms, helping finance teams stay on top of renewals and commitments to reduce compliance risks.
## Getting Started
Setup: Install required libraries and dependencies, and configure OpenAI and FAISS APIs.
Run the Application: Launch the Streamlit app to start uploading, searching, and managing documents.

## Future Enhancements (Phase II)
Enhanced OCR for Scanned Documents: Improved text extraction from more complex scanned documents.
Notifications and Alerts: Automatic notifications for approaching expiration dates.
Integration with Additional Financial Systems: To streamline data import/export with other finance tools.

## License
This project is licensed under the Balanced Rock Power.

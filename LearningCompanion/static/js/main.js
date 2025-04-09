/**
 * Main JavaScript file for the Assessment System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add file input validation
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0].name;
            const fileSize = e.target.files[0].size;
            const fileType = e.target.files[0].type;
            const maxSize = 16 * 1024 * 1024; // 16MB
            
            // Check file size
            if (fileSize > maxSize) {
                alert('File size exceeds the maximum allowed limit (16MB).');
                this.value = '';
                return;
            }
            
            // Check file type
            const allowedTypes = ['application/pdf', 'text/plain'];
            if (!allowedTypes.includes(fileType) && 
                !fileName.endsWith('.pdf') && 
                !fileName.endsWith('.txt')) {
                alert('Only PDF and TXT files are allowed.');
                this.value = '';
                return;
            }
        });
    }

    // Add form validation for document upload
    const uploadForm = document.querySelector('form[action*="upload"]');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const titleInput = document.getElementById('title');
            const fileInput = document.getElementById('file');
            
            if (!titleInput.value.trim()) {
                e.preventDefault();
                alert('Please enter a document title.');
                titleInput.focus();
                return;
            }
            
            if (!fileInput.files[0]) {
                e.preventDefault();
                alert('Please select a file to upload.');
                return;
            }
            
            // Show loading indicator
            document.querySelector('button[type="submit"]').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
            document.querySelector('button[type="submit"]').disabled = true;
        });
    }

    // Auto-refresh the page for documents being processed
    const processingDocuments = document.querySelectorAll('.badge.bg-warning');
    if (processingDocuments.length > 0) {
        setTimeout(function() {
            location.reload();
        }, 30000); // Refresh every 30 seconds
    }

    // Add tab navigation state persistence
    const questionTabs = document.getElementById('questionTabs');
    if (questionTabs) {
        const tabLinks = questionTabs.querySelectorAll('.nav-link');
        tabLinks.forEach(tabLink => {
            tabLink.addEventListener('click', function(e) {
                localStorage.setItem('activeQuestionTab', this.id);
            });
        });
        
        // Restore active tab
        const activeTab = localStorage.getItem('activeQuestionTab');
        if (activeTab) {
            const tabToActivate = document.getElementById(activeTab);
            if (tabToActivate) {
                const tab = new bootstrap.Tab(tabToActivate);
                tab.show();
            }
        }
    }
});

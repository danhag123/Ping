

document.getElementById('queryForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const query = document.getElementById('query').value;
    const chatContainer = document.getElementById('chatContainer');

    // Add user message to chat
    addMessageToChat(query, 'user');
    
    // Clear input field
    document.getElementById('query').value = '';

    try {
        // Show loading indicator
        const loadingIndicator = addMessageToChat('Thinking...', 'bot loading');
        
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        // Remove loading indicator
        chatContainer.removeChild(loadingIndicator);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('API Response:', data);

        // Add bot response to chat
        addMessageToChat(data.answer || data.message || "No response content", 'bot');

    } catch (error) {
        console.error('Error:', error);
        addMessageToChat(`Error: ${error.message}`, 'bot error');
    } finally {
        // Scroll to the bottom of the chat container
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});

// Helper function to add messages
function addMessageToChat(content, className) {
    const chatContainer = document.getElementById('chatContainer');
    const message = document.createElement('div');
    message.className = `message ${className}`;
    message.textContent = content;
    chatContainer.appendChild(message);
    return message;
}
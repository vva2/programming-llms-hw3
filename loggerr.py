import logging
import os.path

if os.path.exists('app.log'):
    os.remove('app.log')

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    filename='app.log',
    force=True
)

# Create a logger for a specific module
logger = logging.getLogger(__name__)
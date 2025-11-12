import os
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from openai import OpenAIError
from dotenv import load_dotenv
from src.bot.rag_pipeline import get_simple_retriever
from src.bot.welcome import WELCOME_MESSAGE
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackContext, filters, CallbackQueryHandler
import re
from difflib import SequenceMatcher
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load environment variables
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define topic keyboard options
TOPIC_KEYBOARD = [
    ["Quiero información general sobre el procedimiento ℹ️"],
    ["Cómo prepararme para la cirugía? 📝"],
    ["Me interesa saber más sobre la rehabilitación y recuperación 🏋️"],
    ["Tengo dudas sobre aspectos logísticos 🏥"],
    ["¿Cómo puedo contactar a mi equipo médico? 📞"],
    ["Estoy nervioso/a y necesito ayuda para calmarme 🫂"]
]

# Topic responses
TOPIC_RESPONSES = {
    "Quiero información general sobre el procedimiento ℹ️": 
        "La artroplastia de rodilla es un procedimiento quirúrgico que reemplaza las superficies dañadas de la "
        "articulación con implantes artificiales. Se realiza cuando hay daño severo por artritis, lesiones u "
        "otras condiciones que causan dolor crónico y limitan la movilidad. Durante la cirugía, el cirujano "
        "reemplaza las superficies dañadas con componentes metálicos y plásticos que simulan el movimiento "
        "natural de la rodilla. ¿Tienes alguna pregunta específica sobre este tema?",
    
    "¿Cómo prepararme para la cirugía? 📝":
        "La preparación para una artroplastia de rodilla incluye evaluaciones médicas completas, ajustes de "
        "medicamentos, planificación de la recuperación en casa, y ejercicios preoperatorios para fortalecer "
        "los músculos. Es importante seguir todas las instrucciones de ayuno antes de la cirugía, organizar "
        "el transporte al hospital y la asistencia posterior, y preparar tu hogar para facilitar la movilidad "
        "reducida durante la recuperación. ¿Tienes alguna pregunta específica sobre la preparación?",

    "Me interesa saber más sobre la rehabilitación y recuperación 🏋️":
        "La rehabilitación después de una artroplastia de rodilla es crucial para el éxito del procedimiento. "
        "Comienza inmediatamente después de la cirugía con ejercicios simples y progresa gradualmente. "
        "Incluye fisioterapia regular, ejercicios en casa, manejo del dolor y seguimiento con el equipo médico. "
        "La recuperación completa puede tomar entre 3 y 6 meses, aunque cada persona avanza a su propio ritmo. "
        "¿Tienes alguna pregunta específica sobre rehabilitación o recuperación?",

    "Tengo dudas sobre aspectos logísticos 🏥":
        "Los aspectos logísticos de una artroplastia de rodilla incluyen aspectos como tiempos en lista de espera, "
        "la planificación de la estancia hospitalaria (generalmente de 1 a 3 días), la organización del transporte "
        "y la preparación del hogar para el regreso. También es importante coordinar las citas de seguimiento y "
        "sesiones de fisioterapia, y considerar si necesitarás dispositivos de asistencia como andadores o "
        "bastones. ¿Tienes alguna pregunta específica sobre estos aspectos logísticos?",

    "¿Cómo puedo contactar a mi equipo médico? 📞":
        "Para contactar a tu equipo médico, lo mejor es llamar al número de teléfono proporcionado por tu clínica, este es el telefono de contacto "
        "000-000-0000. También puedes enviar un correo electrónico al 0000@ejemplo.com. Si tienes una aplicación "
        "de salud o portal en línea, allí también podrías encontrar información de contacto y la opción de enviar "
        "mensajes directos a tu equipo. Es importante que tengas a mano tu información médica y cualquier pregunta "
        "específica que quieras discutir. ¿Necesitas ayuda para encontrar el contacto correcto?",
    
    "Estoy nervioso/a y necesito ayuda para calmarme 🫂":
        "Entiendo que esta situación puede generar miedo o ansiedad, y es completamente normal sentirse así antes de una cirugía. "
        "Estoy aquí para ayudarte. ¿Te gustaría compartir cómo te sientes en este momento? O puedo ofrecerte algunas sugerencias para manejar la ansiedad, como:\n\n"
        "• *Información sobre el procedimiento* para que sepas qué esperar y te sientas más tranquilo/a 📖\n"
        "• *Técnicas de relajación* (respiración, mindfulness) 🧘\n"
        "• *Qué esperar el día de la cirugía* para sentirte más preparado/a 📅\n"
        "• *Testimonios de otros pacientes* que pasaron por esto 💙\n\n"
        "O prefieres hablar de otro tema estoy disponible para ti.",
}

# PromptTemplate para empoderar RAG + empatía
template = """
Eres un asistente virtual empático y confiable diseñado para ayudar a pacientes que se han sometido o se someterán a una cirugía de artroplastia de rodilla.
Tu objetivo es responder de forma clara, respetuosa y tranquilizadora, basándote en la información contenida en los documentos que tienes como referencia.
Si la pregunta se relaciona con el proceso quirúrgico, la recuperación, el dolor, la fisioterapia o las emociones asociadas al procedimiento, intenta brindar orientación general y apoyo,
incluso si no cuentas con información específica exacta. Sé honesto si no tienes una respuesta precisa, pero ofrece siempre una alternativa útil, como consultar al equipo médico. Si no entiendes la pregunta, pide aclaraciones de manera amable.
Si el usuario menciona síntomas específicos, como dolor, inflamación o fiebre, sugiere que consulte a su médico o equipo de salud para una evaluación adecuada.
Si detectas ansiedad, miedo o dolor en el usuario, valida sus emociones y responde con calidez, sin sonar automatizado.
Nunca inventes datos médicos ni ofrezcas diagnósticos personalizados.

Tus respuestas deben ser cortas (de 1.5 párrafos en promedio), sencillas de entender y empáticas, evitando tecnicismos innecesarios.

Si el usuario cambia de tema, saluda o hace una broma, debes responder al nuevo mensaje sin repetir lo anterior. 
Si el usuario pregunta por temas no relacionados con la cirugía de rodilla, como deportes, hobbies o temas generales, responde de manera amigable y breve, pero sin desviarte del tema principal de la cirugía de rodilla, no des información sobre otros temas.

ofrece más información despues de responder a la pregunta principal, pero no te extiendas demasiado.

Si no puedes responder con certeza, puedes decir: 'No tengo información específica sobre eso, pero te recomiendo hablar con tu equipo médico para recibir orientación precisa.'

Usa la siguiente información de contexto para responder en español la pregunta, respuestas de 1.5 párrafos en promedio:
{context}

Pregunta: {question}
Respuesta útil:"""

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Métricas de evaluación en tiempo real
class ChatbotEvaluator:
    def __init__(self):
        self.total = 0
        self.abstentions = 0

    def record(self, response: str):
        self.total += 1
        if "No tengo información específica" in response:
            self.abstentions += 1

    def stats(self):
        if self.total == 0:
            return {"tasa_abstencion": 0.0}
        return {"tasa_abstencion": self.abstentions / self.total}

evaluator = ChatbotEvaluator()

def create_chatbot():
    """
    Configura el chatbot con LLaMA 3 vía Groq, RAG pipeline y memoria conversacional.
    """
    # Ensure we're using the correct API key (not from cached environment)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY no está definida en las variables de entorno")
    
    # Configurar el modelo LLM con los parámetros correctos
    logger.info("🤖 Configurando el modelo LLM...")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.4,
        max_tokens=500
    )
    
    # Verificar que el modelo se configuró correctamente
    logger.info(f"✓ Modelo configurado: {llm.model_name}")
    
    # Test del modelo
    try:
        logger.info("🔄 Probando conexión con el modelo...")
        test_response = llm.invoke("Test connection")
        logger.info("✓ Prueba de modelo exitosa")
    except Exception as e:
        logger.error(f"✗ Error al probar el modelo: {e}")
        raise

    # Configurar memoria
    logger.info("🧠 Configurando memoria conversacional...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Get the retriever
    logger.info("📚 Configurando sistema de recuperación...")
    retriever = get_simple_retriever()
    
    # Crear el chain con configuración personalizada
    logger.info("🔗 Creando cadena conversacional...")
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        },
        verbose=True  # Habilitar logs detallados
    )
    return convo_chain

def generate_response(chain, user_input: str):
    """
    Genera una respuesta usando el chain conversacional con manejo de errores.
    """
    try:
        output = chain.invoke({"question": user_input})
        result = output.get("answer", "")
        source_documents = output.get("source_documents", [])
    except Exception as e:
        if "Invalid API Key" in str(e) or "401" in str(e):
            logger.error("Error: Groq API Key inválida o expirada")
            result = (
                "🔧 **Estado del Bot: Modo de Mantenimiento**\n\n"
                "Actualmente estamos experimentando problemas técnicos con nuestro servicio de IA. "
                "El sistema de recuperación de documentos está funcionando correctamente, pero la generación de respuestas está temporalmente deshabilitada.\n\n"
                "**Mientras tanto, puedes:**\n"
                "• Contactar a tu equipo médico directamente\n"
                "• Revisar la documentación proporcionada por tu clínica\n"
                "• Intentar nuevamente en unos minutos\n\n"
                "Lamentamos las molestias. Estamos trabajando para resolver este problema."
            )
        else:
            logger.error(f"Unexpected error: {e}", exc_info=e)
            result = (
                "Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo."
            )
        source_documents = []
    
    evaluator.record(result)
    return result, source_documents

async def start_command(update: Update, context: CallbackContext):
    """Handler para el comando /start"""
    # Reset memory for new conversation
    context.bot_data["chain"].memory.clear()
    
    # Create reply keyboard with suggested topics
    reply_markup = ReplyKeyboardMarkup(
        TOPIC_KEYBOARD,
        resize_keyboard=True,
        one_time_keyboard=False
    )
    
    # Send welcome message with keyboard
    await update.message.reply_text(
        WELCOME_MESSAGE,
        reply_markup=reply_markup
    )
    logger.info(f"Nuevo usuario o sesión iniciada: {update.effective_user.username or update.effective_user.id}")

async def handle_topic_selection(update: Update, context: CallbackContext):
    """Handler específico para cuando el usuario selecciona uno de los temas sugeridos"""
    user_input = update.message.text
    
    # Check if the input matches one of our predefined topics
    if user_input in TOPIC_RESPONSES:
        # Send the predefined response for the selected topic
        await update.message.reply_text(TOPIC_RESPONSES[user_input])
        # Log that a topic was selected
        logger.info(f"Usuario seleccionó tema: {user_input}")
        return True
    return False

async def handle_message(update: Update, context: CallbackContext):
    """Handler asíncrono para mensajes de Telegram"""
    user_input = update.message.text
    
    # First check if it's a topic selection
    is_topic = await handle_topic_selection(update, context)
    if is_topic:
        return
    
    # If not a predefined topic, process as regular input
    result, sources = generate_response(context.bot_data["chain"], user_input)
    
    # Format the response with sources
    if sources:
        # Get unique sources to avoid duplicates
        unique_sources = list(set([doc.metadata.get('source', 'Fuente no disponible') for doc in sources]))
        
        # Add sources to the response
        sources_text = "\n\n📚 **Fuentes consultadas:**\n"
        for i, source in enumerate(unique_sources, 1):
            # Clean up the source name (remove file extensions and format nicely)
            clean_source = source.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            sources_text += f"• {clean_source}\n"
        
        # Combine response with sources
        full_response = result + sources_text
    else:
        full_response = result + "\n\n📚 **Fuentes consultadas:** Respuesta basada en conocimiento general"
    
    # Handle message length limits (Telegram has a 4096 character limit)
    max_length = 3900  # Safe limit accounting for formatting
    
    if len(full_response) > max_length:
        # Split on paragraph boundaries when possible
        paragraphs = full_response.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If a single paragraph is too long, split it
            if len(paragraph) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                # Split the long paragraph into smaller chunks
                para_chunks = [paragraph[i:i+max_length] for i in range(0, len(paragraph), max_length)]
                chunks.extend(para_chunks)
            else:
                # Check if adding this paragraph would exceed max_length
                if len(current_chunk) + len(paragraph) + 2 > max_length:  # +2 for "\n\n"
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Send the chunks
        for i, chunk in enumerate(chunks):
            if i == 0:
                await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(f"(Continuación {i+1}/{len(chunks)})\n{chunk}")
    else:
        await update.message.reply_text(full_response)

async def reset_conversation(update: Update, context: CallbackContext):
    """Reinicia la memoria de conversación cuando el usuario usa el comando /reset"""
    context.bot_data["chain"].memory.clear()
    logger.info(f"Memoria reiniciada por usuario: {update.effective_user.username or update.effective_user.id}")
    
    # Re-display the keyboard with topic options
    reply_markup = ReplyKeyboardMarkup(
        TOPIC_KEYBOARD,
        resize_keyboard=True,
        one_time_keyboard=False
    )

    await update.message.reply_text(
        "🔄 Memoria de conversación reiniciada. ¿En qué puedo ayudarte hoy?",
        reply_markup=reply_markup
    )

# This function is not needed as we're using ReplyKeyboardMarkup which is handled by handle_message
# If we switch to InlineKeyboardMarkup in the future, we would use this approach

def main():
    """Función principal para iniciar el bot"""
    # Crea el chatbot y guárdalo en bot_data para usar en handlers
    chain = create_chatbot()
    
    # Limpiar la memoria al iniciar el bot para asegurar conversaciones nuevas
    logger.info("🧹 Limpiando memoria de conversaciones anteriores...")
    chain.memory.clear()

    # Inicializa el bot de Telegram (v20+ syntax) - Using TELEGRAM_TOKEN2 for Groq version
    application = Application.builder().token(os.environ["TELEGRAM_TOKEN2"]).build()
    application.bot_data["chain"] = chain

    # Handlers para comandos y mensajes
    application.add_handler(CommandHandler("reset", reset_conversation))
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Send welcome message on startup to active chats
    async def send_welcome_on_startup(context: CallbackContext):
        """Send welcome message to recent users on bot startup"""
        try:
            # Initialize the bot if not already initialized
            if not context.bot._initialized:
                await context.bot.initialize()
            
            # Get recent updates to find active chats
            updates = await context.bot.get_updates(limit=100)
            sent_to = set()  # Track chat IDs to avoid duplicates
            
            logger.info("🔄 Enviando mensaje de bienvenida a usuarios recientes...")
            
            for update in updates:
                chat_id = None
                if update.message:
                    chat_id = update.message.chat_id
                elif update.callback_query:
                    chat_id = update.callback_query.message.chat_id
                
                if chat_id and chat_id not in sent_to:                    # Create topic keyboard
                    reply_markup = ReplyKeyboardMarkup(
                        TOPIC_KEYBOARD,
                        resize_keyboard=True,
                        one_time_keyboard=False
                    )
                    
                    # Send restart notification and welcome message with keyboard
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="🔄 *El bot ha sido reiniciado*\nTu historial de conversación anterior ha sido borrado.\n\n" + WELCOME_MESSAGE,
                        reply_markup=reply_markup
                    )
                    sent_to.add(chat_id)
                    logger.info(f"Mensaje de reinicio enviado al chat {chat_id}")
            
            if not sent_to:
                logger.info("No se encontraron chats activos recientes.")
        except Exception as e:
            logger.error(f"Error al enviar mensajes de reinicio: {e}")
    
    # Schedule the welcome message to be sent right after startup
    application.job_queue.run_once(send_welcome_on_startup, 0)
    
    logger.info("🤖 Chatbot listo. Iniciando polling...")
    application.run_polling()
    
    # Al finalizar, mostrar métricas
    stats = evaluator.stats()
    logger.info(f"Tasa de abstención: {stats['tasa_abstencion']*100:.1f}%")

def run_chatbot():
    """Función wrapper para ejecutar el bot desde app2.py"""
    main()

if __name__ == "__main__":
    main()
# This code is a Telegram bot that uses a conversational retrieval chain to answer questions about knee arthroplasty surgery.

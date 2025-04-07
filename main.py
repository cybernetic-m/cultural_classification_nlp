from neo4j import GraphDatabase
#from neo4jconn import Neo4jConnection


NEO4J_URI="neo4j+s://28b54ee4.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="VI0I_AyGCe2bLsaSdBRKYESrsw063MhWVHsmQxlAVvk"
AURA_INSTANCEID="28b54ee4"
AURA_INSTANCENAME="Free instance"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def test_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Connessione riuscita!' AS messaggio")
            print(result.single()["messaggio"])
    except Exception as e:
        print("Errore nella connessione:", e)


def crea_nodo_demo():
    with driver.session() as session:
        session.run("CREATE (:Persona {nome: 'Alice', eta: 30})")
        print("Nodo 'Alice' creato!")


test_connection()
crea_nodo_demo()


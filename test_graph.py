import asyncio
import re
from app.db.neo4j_client import init_neo4j, close_neo4j

async def main():
    driver = await init_neo4j()
    async with driver.session() as session:
        res = await session.run('SHOW CONSTRAINTS')
        constrs = await res.data()
        print('CONSTRAINTS:', constrs)
        
        for c in constrs:
            name = c.get('name')
            # Sanitize: only allow alphanumeric and underscores in constraint names
            if not re.match(r'^[a-zA-Z0-9_]+$', name):
                print(f'Skipping unsafe constraint name: {name}')
                continue
            print(f'Dropping constraint: {name}')
            await session.run(f'DROP CONSTRAINT {name}')
            
    await close_neo4j()

if __name__ == '__main__':
    asyncio.run(main())

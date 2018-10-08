import scrap

scrap.sayK('Gill')


#File System
fo = open('test.txt', 'w')

print('Name: ', fo.name)
print('Is closed: ', fo.closed)
print('Opening Mode: ', fo.mode)
fo.write('I love poon')



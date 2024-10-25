!pip install metadata_parser
import metadata_parser as mp

url_india='https://www.indiacitywalks.com/'
url_cloq='https://cloq.app/'
url_surai='https://www.suraitech.in/'
url_teplu='https://teplu.in/'
url_prayogik='https://prayogik.in/'

page_india = mp.MetadataParser(url_india)
page_cloq = mp.MetadataParser(url_cloq)
page_surai = mp.MetadataParser(url_surai)
page_teplu = mp.MetadataParser(url_teplu)
page_prayogik = mp.MetadataParser(url_prayogik)

meta_data_india = page_india.get_metadatas('description')[0]
meta_data_cloq = page_cloq.get_metadatas('description')[0]
meta_data_teplu = page_teplu.get_metadatas('description')[0]
meta_data_prayogik = page_prayogik.get_metadatas('description')[0]

print("IndiaCityWalks: ", meta_data_india)
print("Cloq: ", meta_data_cloq)
print("Teplu: ", meta_data_teplu)
print("Prayogik: ", meta_data_prayogik)
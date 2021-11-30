import glob

my_path = "\\nas.streetscan.com/bulk_dev/MA_Burlington/011/20190418_MA_Burlington_Undef_1/0"

files = glob.iglob(my_path + '/**/*.meta', recursive=True)
print(files)


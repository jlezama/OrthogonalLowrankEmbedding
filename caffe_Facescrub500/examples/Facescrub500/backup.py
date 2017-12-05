import os, zipfile





def findExt(folder, extensions, exclude_list):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        if any(substring in root for substring in exclude_list):
            # don't include result
            continue
        for extension in extensions:
            for filename in filenames:
                if filename.endswith(extension):
                    # print root, filename
                    matches.append(os.path.join(root, filename))

    return matches

def backup_code(outfname, folder, extensions, exclude_list):
    filenames = findExt(folder, extensions, exclude_list)
    # print filenames
    zf = zipfile.ZipFile(outfname, mode='w')
    for filename in filenames:
        zf.write(filename)
    zf.close()
    print 'saved %i files to %s' % (len(filenames), outfname)



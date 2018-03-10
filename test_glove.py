import glove

cooccur = {
	0: {
		0: 1.0,
		2: 3.5
	},
	1: {
		2: 0.5
	},
	2: {
		0: 3.5,
		1: 0.5,
		2: 1.2
	}
}

cooccur_mat = [[1,2,3],[2,1,2],[2,1,3]]

# convert matrix to dict
keys = range(len(cooccur_mat))
lines = [ ]
for i in range(len(cooccur_mat)):
     line = dict(zip(keys, cooccur_mat[i]))
     lines.append(line)

c_c_mat = dict(zip(keys, lines))

model = glove.Glove(c_c_mat, d=50, alpha=0.75, x_max=100.0)

for epoch in range(25):
    err = model.train(step_size = 0.05, workers=9, batch_size=50)
    print err

print model.W
print model.b 

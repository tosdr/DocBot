import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*continuing)|(?=.*continued))(?=.*use)(((?=.*accept)|(?=.*agree))((?=.*terms)|(?=.*change)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 403,
	name: "Instead of asking directly, this Service will assume your consent merely from your usage."
} as Regex;
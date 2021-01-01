import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*interest-based ads)|(?=.*targeted advertising))((?=.*opt)|((?=.*set)(?=.*preference)))", "i"),
	caseID: 219,
	name: "You can opt out of targeted advertising"
} as Regex;